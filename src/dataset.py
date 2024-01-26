"""Dataset utilities"""
from __future__ import annotations

import atexit
import logging
import re
import shutil
import sys
from csv import DictReader
from multiprocessing import Event, JoinableQueue, Process, cpu_count
from pathlib import Path
from queue import Empty
from typing import Dict, NoReturn, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from tqdm import tqdm

from config import settings


def _image_processor(process_queue: JoinableQueue, shutdown: Event) -> NoReturn:
    """Consumer process
    Load, resize, normalize and save image

    Parameters
    ----------
    process_queue: Queue
        Message queue with image tuple path (source, destination) to process
    shutdown: Event
        Shutdown notification event. If queue is empty and event set exit from process

    References
    ----------
    Yaman, S., Karakaya, B. & Erol, Y.
     A novel normalization algorithm to facilitate pre-assessment of Covid-19 disease by improving accuracy of CNN
     and its FPGA implementation. Evolving Systems (2022). https://doi.org/10.1007/s12530-022-09419-3
    """
    image_size = settings.preprocessing.image_size
    while True:
        try:
            img_source, img_destination = process_queue.get(block=True, timeout=1)
        except Empty:
            if shutdown.is_set():
                sys.exit(0)
            continue
        # Open
        img = Image.open(img_source)
        # Resize
        resized_img = img.resize(size=image_size, resample=Image.LANCZOS).convert(mode="L")
        # Normalize
        image_array = np.asarray(resized_img)
        normalized = (image_array - np.mean(image_array)) / np.std(image_array)
        scaled = (normalized - np.min(normalized)) / (np.max(normalized) - np.min(normalized))
        scaled = scaled * 255
        norm_img = Image.fromarray(scaled.astype(np.int8), mode="L")
        # Save
        norm_img.save(img_destination, icc_profile=False)
        # Notify
        process_queue.task_done()


def _pre_processes_image() -> None:
    """Prepare dataset.
    Scan over CSV file and copy images into cooked folder per class.
    Images resized and normalized
    """
    logger = logging.getLogger("raido")

    logger.info("Removing all existing data in cooked folder - %s", settings.folders.cooked_folder)
    shutil.rmtree(settings.folders.cooked_folder, ignore_errors=True)
    settings.folders.cooked_folder.mkdir(parents=True)
    for diagnosis in settings.preprocessing.target_diagnosis:
        (settings.folders.cooked_folder / diagnosis).mkdir()

    logger.info("Starting processing workers")
    work_queue = JoinableQueue(cpu_count() * 2 + 1)
    exit_event = Event()
    workers = [
        Process(
            target=_image_processor,
            kwargs={"process_queue": work_queue, "shutdown": exit_event},
        )
        for _ in range(cpu_count() // 2 + 1)
    ]
    atexit.register(exit_event.set)  # Fail-safe
    for worker in workers:
        worker.start()
    total_images = sum(1 for _ in settings.folders.raw_folder.glob("*.png"))
    logger.info("Loading files. Total files - %s", total_images)
    with open(settings.folders.root_folder / "Data_Entry.csv", "r", encoding="utf-8") as csv_fh:
        reader = DictReader(csv_fh)
        for row in tqdm(reader, desc="Preprocessing files", miniters=0, unit="images", total=total_images):
            detected_diagnosis = [
                diagnosis
                for diagnosis in row["Finding Labels"].split("|")
                if diagnosis in settings.PREPROCESSING.target_diagnosis
            ]
            if len(detected_diagnosis) != 1:
                continue  # Ignore multi-labeled images
            diagnosis_label = detected_diagnosis[0].replace("_", " ")
            target_folder = settings.folders.cooked_folder / diagnosis_label
            target_name = (
                f"{row['Patient Gender'].upper()}_{int(row['Patient Age']):03d}"
                f"_{diagnosis_label}_{row['Image Index']}"
            )
            work_queue.put(
                (settings.folders.raw_folder / row["Image Index"], target_folder / target_name), block=True, timeout=10
            )

        logger.info("Parsing finished. Waiting for workers")
        exit_event.set()
        for worker in workers:
            worker.join()


def _scan_cooked_dataset(only_male: bool) -> Dict[str, Tuple[Path, ...]]:
    """Scan preprocessed dataset and return diagnosis and list of images file path

    Parameters
    ----------
    only_male: bool
        If set return only male patients, else only female patients,

    Returns
    -------
    Dict[str, Tuple[Path, ...]]
        Dictionary where key is diagnosis and value is tuple of images path according to scan options
    """
    logger = logging.getLogger("raido")
    results = {}
    for diagnosis in settings.PREPROCESSING.target_diagnosis:
        if (settings.folders.cooked_folder / diagnosis).exists():
            logger.info("Scanning folder - %s", (settings.folders.cooked_folder / diagnosis))
            scan_results = []
            for _image in (settings.folders.cooked_folder / diagnosis).glob("*.png"):
                if only_male is True and _image.name.startswith("M"):
                    scan_results.append(_image)
                elif only_male is False and _image.name.startswith("F"):
                    scan_results.append(_image)
            scan_results.sort()
            results[diagnosis] = tuple(scan_results)
    return results


def _create_validation_set() -> None:
    """Create a validation set from cooked data"""
    logger = logging.getLogger("raido")
    rng = np.random.default_rng(seed=settings.dataset.seed)

    logger.info(
        "Removing all existing data in validation folder - %s",
        settings.folders.validation_folder,
    )
    shutil.rmtree(settings.folders.validation_folder, ignore_errors=True)
    settings.folders.validation_folder.mkdir()

    for test_group in ["Male", "Female"]:
        source_images = _scan_cooked_dataset(only_male=test_group == "Male")
        for diagnosis in settings.preprocessing.target_diagnosis:
            logger.info("Creating sub-folder for diagnosis %s", diagnosis)
            diagnosis_folder = settings.folders.validation_folder / diagnosis
            diagnosis_folder.mkdir(exist_ok=True)

            selected_images = rng.choice(
                source_images[diagnosis],
                size=int(len(source_images[diagnosis]) * settings.dataset.validation_fraction),
                replace=False,
            )
            for image_path in selected_images:
                shutil.move(image_path, diagnosis_folder)


def _create_test_set() -> None:
    """Create a test set from cooked data"""
    logger = logging.getLogger("raido")
    rng = np.random.default_rng(seed=settings.dataset.seed)

    logger.info("Removing all existing data in test folder - %s", settings.folders.test_folder)
    shutil.rmtree(settings.folders.test_folder, ignore_errors=True)
    settings.folders.test_folder.mkdir()
    uni_folder = settings.folders.test_folder / "Uni"
    logger.info("Creating test dataset with both groups - %s", uni_folder)
    uni_folder.mkdir()

    for test_group in ["Male", "Female"]:
        group_folder = settings.folders.test_folder / test_group
        logger.info("Creating test dataset with specific group - %s", group_folder)
        group_folder.mkdir()
        source_images = _scan_cooked_dataset(only_male=test_group == "Male")
        for diagnosis in settings.preprocessing.target_diagnosis:
            logger.info("Creating sub-folder for diagnosis %s", diagnosis)
            diagnosis_folder = group_folder / diagnosis
            diagnosis_folder.mkdir(exist_ok=True)
            uni_diagnosis_folder = uni_folder / diagnosis
            uni_diagnosis_folder.mkdir(exist_ok=True)

            selected_images = rng.choice(source_images[diagnosis], size=settings.dataset.test_images, replace=False)
            for image_path in selected_images:
                shutil.copy(image_path, uni_diagnosis_folder)
                shutil.move(image_path, diagnosis_folder)


def _undersample_dataset() -> None:
    """Count images by diagnosis and sex and reduce dataset size to the lowest number"""
    logger = logging.getLogger("raido")
    rng = np.random.default_rng(seed=settings.dataset.seed)

    logger.info("Scanning for images count")
    male_source_images = _scan_cooked_dataset(only_male=True)
    female_source_images = _scan_cooked_dataset(only_male=False)
    lowest = float("inf")
    for _items in list(male_source_images.values()) + list(female_source_images.values()):
        if lowest > len(_items):
            lowest = len(_items)
    logger.info("Lowest image group is %s", lowest)
    for diagnosis in settings.preprocessing.target_diagnosis:
        for group, source_images in [
            ("Male", male_source_images[diagnosis]),
            ("Female", female_source_images[diagnosis]),
        ]:
            uncorrupted_images = [image for image in source_images if diagnosis.lower() in image.name.lower()]
            images_to_remove = len(source_images) - lowest
            if images_to_remove > len(uncorrupted_images):
                logger.info("Padding uncorrupted images")
                corrupted = [image for image in source_images if diagnosis.lower() not in image.name.lower()]
                uncorrupted_images += list(
                    rng.choice(corrupted, size=images_to_remove - len(uncorrupted_images), replace=False)
                )

            logger.info(
                "Downsampling %s - %s patients. Removing %s images",
                diagnosis,
                group,
                images_to_remove,
            )
            for image_path in rng.choice(uncorrupted_images, size=images_to_remove, replace=False):
                image_path.unlink()


def _reduce_dataset(fraction: float, only_male: Union[bool, None] = None) -> None:
    """Reduce dataset by fraction [0-1] relative to selected group

    Parameters
    ----------
    fraction: float
        Fraction to reduce dataset
    only_male: Union[bool, None]
        Reduce only specific group. Set - male patients, Clear - female patients, None - both
    """
    logger = logging.getLogger("raido")
    rng = np.random.default_rng(seed=settings.dataset.seed)

    if fraction <= 0:
        logger.info("No reduction required. Exit")
        return

    logger.info("Scanning for images count")
    male_source_images = _scan_cooked_dataset(only_male=True)
    female_source_images = _scan_cooked_dataset(only_male=False)

    for diagnosis in settings.preprocessing.target_diagnosis:
        if only_male is True or only_male is None:
            images_to_delete = int(len(male_source_images[diagnosis]) * fraction)
            for image_path in rng.choice(male_source_images[diagnosis], size=images_to_delete, replace=False):
                image_path.unlink()
        elif only_male is False or only_male is None:
            images_to_delete = int(len(female_source_images[diagnosis]) * fraction)
            for image_path in rng.choice(female_source_images[diagnosis], size=images_to_delete, replace=False):
                image_path.unlink()


def _corrupt_images(fraction: float, only_male: bool) -> None:
    """Corrupt diagnosis
    Randomly select images and mark them as `keep_diagnosis`

    Parameters
    ----------
    fraction: float
        Fraction to corrupt dataset labels
    only_male: Union[bool, None]
        Corrupt only specific group. Set - male patients, Clear - female patients, None - both
    """
    logger = logging.getLogger("raido")
    rng = np.random.default_rng(seed=settings.dataset.seed)
    target_folder = settings.folders.cooked_folder / settings.preprocessing.keep_diagnosis

    if fraction <= 0:
        logger.info("No corruption required. Exit")
        return

    logger.info("Scanning for images count")
    male_source_images = _scan_cooked_dataset(only_male=True)
    female_source_images = _scan_cooked_dataset(only_male=False)
    lowest = float("inf")
    for _items in list(male_source_images.values()) + list(female_source_images.values()):
        if lowest > len(_items):
            lowest = len(_items)

    images_to_delete = int(lowest * fraction)

    for diagnosis in settings.preprocessing.target_diagnosis:
        if diagnosis == settings.preprocessing.keep_diagnosis:
            continue

        if only_male is True or only_male is None:
            for image_path in rng.choice(male_source_images[diagnosis], size=images_to_delete, replace=False):
                shutil.move(image_path, target_folder)
        elif only_male is False or only_male is None:
            for image_path in rng.choice(female_source_images[diagnosis], size=images_to_delete, replace=False):
                shutil.move(image_path, target_folder)


def create_dataset(corruption: float, reduction: float, only_male: Union[bool, None]) -> Tuple[float, float]:
    """Create dataset for training/validation and test

    Parameters
    ----------
    corruption: float
        Corruption fraction
    reduction: float
        Reduction fraction
    only_male: Union[bool, None]
        Set apply corruption and reduction to male patients only, Clear to female, None to both

    Returns
    -------
    Tuple[float, float]
        Actual corruption and reduction fraction
    """
    logger = logging.getLogger("raido")
    logger.info("Creating dataset")
    _pre_processes_image()

    logger.info("Creating test set")
    _create_test_set()

    logger.info("Applying label corruption")
    _corrupt_images(fraction=corruption, only_male=only_male)

    logger.info("Under-sampling dataset")
    _undersample_dataset()

    logger.info("Calculating total number of images")
    total_images = 0
    for diagnosis in settings.PREPROCESSING.target_diagnosis:
        for _image in (settings.folders.cooked_folder / diagnosis).glob("*.png"):
            total_images += 1
    logger.info("Total number of images before reduction - %s", total_images)
    logger.info("Applying reduction")
    _reduce_dataset(fraction=reduction, only_male=only_male)
    reduced_images = 0
    for diagnosis in settings.PREPROCESSING.target_diagnosis:
        for _image in (settings.folders.cooked_folder / diagnosis).glob("*.png"):
            reduced_images += 1

    logger.info("Calculating total number of images")
    corrupted = 0
    correct = 0
    for _image in (settings.folders.cooked_folder / settings.preprocessing.keep_diagnosis).glob("*.png"):  # type: Path
        if settings.preprocessing.keep_diagnosis.lower() in _image.name.lower():
            correct += 1
        else:
            corrupted += 1
    measured_corruption = (corrupted / (correct + corrupted)) * 100
    measured_reduction = 100 - reduced_images / total_images * 100
    logger.info(
        "Measured reduction - %05.1f%%, measured corruption - %05.1f%%", measured_reduction, measured_corruption
    )

    logger.info("Creating validation dataset")
    _create_validation_set()

    return measured_corruption, measured_reduction


def get_train_dataset(batch_size: int) -> DataLoader:
    """Return train dataset

    Parameters
    ----------
    batch_size: int
        Batch size

    Returns
    -------
    DataLoader
        Train dataset
    """
    logger = logging.getLogger("raido")

    logger.info("Loading train dataset")
    affine = v2.RandomAffine(
        degrees=settings.augmentation.rotation_angle,
        scale=[1.0 - settings.augmentation.zoom_range, 1.0 + settings.augmentation.zoom_range],
        translate=[settings.augmentation.width_shift_range, settings.augmentation.height_shift_range],
    )
    train_transforms = v2.Compose(
        [affine, v2.Grayscale(num_output_channels=1), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    )
    train_dataset = ImageFolder(root=f"{settings.folders.cooked_folder}", transform=train_transforms)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=settings.dataset.num_of_worker,
        pin_memory=True,
        drop_last=True,
    )
    return train_dataloader


def get_validation_dataset(batch_size: int) -> DataLoader:
    """Return validation dataset

    Parameters
    ----------
    batch_size: int
        Batch size

    Returns
    -------
    DataLoader
        Train dataset
    """
    logger = logging.getLogger("raido")

    logger.info("Loading validation dataset")
    data_transform = v2.Compose(
        [v2.Grayscale(num_output_channels=1), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    )
    val_dataset = ImageFolder(root=f"{settings.folders.validation_folder}", transform=data_transform)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=settings.dataset.num_of_worker,
        pin_memory=True,
        drop_last=True,
    )
    return val_dataloader


def get_test_dataset(only_male: Union[bool, None], batch_size: int) -> DataLoader:
    """Return test dataset

    Parameters
    ----------
    only_male: Union[bool, None]
        If set dataset will contain only male patients, clear only female patients, None both
    batch_size: int
        Batch size

    Returns
    -------
    tf.data.Dataset
        Test dataset
    """
    logger = logging.getLogger("raido")

    if only_male is True:
        test_dataset_folder = settings.folders.test_folder / "Male"
    elif only_male is False:
        test_dataset_folder = settings.folders.test_folder / "Female"
    else:
        test_dataset_folder = settings.folders.test_folder / "Uni"

    logger.info("Loading test dataset from folder - %s", test_dataset_folder)
    data_transform = v2.Compose(
        [v2.Grayscale(num_output_channels=1), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    )
    test_dataset = ImageFolder(root=f"{test_dataset_folder}", transform=data_transform)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=settings.dataset.num_of_worker,
        pin_memory=False,
        drop_last=False,
    )
    return test_dataloader


def get_sample_images() -> Tuple[Path, ...]:
    """Return sample images path

    Returns
    -------
    Tuple[Path, ...]
        Female diagnosis, Male diagnosis
    """
    images = []
    for row_sex in ["Female", "Male"]:
        for col_diagnosis in sorted(settings.preprocessing.target_diagnosis):
            img_path = list(Path(settings.folders.test_folder, "Uni", f"{col_diagnosis}").glob("*.png"))
            img_name = sorted([img.name for img in img_path])
            re_filter = re.compile(f"{row_sex[0]}" + r"_0(1[8-9]|[2-9]{2}).+\.png")  # Skip small images
            selected_img = list(filter(re_filter.match, img_name))[0]
            images.append(Path(settings.folders.test_folder, "Uni", f"{col_diagnosis}", selected_img))
    return tuple(images)
