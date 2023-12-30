"""Pre-process dataset"""
from __future__ import annotations

import atexit
import logging
import shutil
import sys
from csv import DictReader
from multiprocessing import Event, JoinableQueue, Process, cpu_count
from pathlib import Path
from queue import Empty
from time import monotonic
from typing import Dict, NoReturn, Tuple

import numpy as np
from PIL import Image
from sqlalchemy import func

from by_age.database import Diagnosis, Images, get_diagnosis_list, session_scope
from by_age.visualization import plot_labels, plot_rebalanced_dataset
from config import PreProcessing


def cook_dataset() -> None:
    """Cook dataset main function"""
    logger = logging.getLogger("raido")
    logger.info("Pre-processing dataset")
    with session_scope() as session:
        session.query(Images).delete()
        session.query(Diagnosis).delete()
    pre_processes_image()
    logger.info("Under-sampling dataset")
    calculate_average_age()
    undersample_dataset()
    logger.info("Plotting dataset statistics")
    export_path = Path("reports", "dataset")
    export_path.mkdir(parents=True, exist_ok=True)
    logger.info("Plotting dataset statistics")
    plot_labels(export_path=export_path)
    plot_rebalanced_dataset(export_path=export_path)


def calculate_average_age():
    """Set is_above metric based on average age"""
    logger = logging.getLogger("raido.dataset")
    diagnosis_average = {}
    with session_scope() as session:
        for diagnosis in get_diagnosis_list(session):
            diagnosis_average[diagnosis] = (
                session.query(func.avg(Images.age))
                .join(Diagnosis, Images.diagnosis)
                .filter(Diagnosis.diagnosis == diagnosis)
            )[0][0]
    logger.info("Diagnosis average ages: %s", diagnosis_average)
    with session_scope() as session:
        for image in session.query(Images).filter(Images.ignored.is_(False)):  # type: Images
            image.is_above = image.age >= diagnosis_average[image.diagnosis.diagnosis]


def image_processor(process_queue: JoinableQueue, shutdown: Event) -> NoReturn:
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
    _settings = PreProcessing()
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
        resized_img = img.resize(size=_settings.image_size, resample=Image.LANCZOS).convert(mode="L")
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


# pylint: disable=too-many-locals
def pre_processes_image() -> None:
    """Prepare dataset.
    Scan over CSV file and insert images into database. Images are resized and normalized
    """
    logger = logging.getLogger("raido.dataset")
    cooked_folder = Path("data", "cooked").absolute()
    logger.info("Removing all existing data in cooked folder - %s", cooked_folder)
    raw_folder = Path("data", "raw").absolute()
    shutil.rmtree(cooked_folder, ignore_errors=True)
    cooked_folder.mkdir(parents=True, exist_ok=True)
    logger.info("Starting Image processing workers")
    work_queue = JoinableQueue(100)
    exit_event = Event()
    workers = [
        Process(target=image_processor, kwargs={"process_queue": work_queue, "shutdown": exit_event})
        for _ in range(cpu_count() // 2 + 1)
    ]
    atexit.register(exit_event.set)  # Fail-safe
    for worker in workers:
        worker.start()
    logger.info("Processing all images")
    sub_total = 0
    known_diagnosis = {}  # type: Dict[str, int]
    run_time = monotonic()
    with open(raw_folder.parent / "Data_Entry.csv", "r", encoding="utf-8") as csv_fh:
        reader = DictReader(csv_fh)
        with session_scope() as session:
            for row_count, row in enumerate(reader):
                detected_diagnosis = [
                    diagnosis
                    for diagnosis in row["Finding Labels"].split("|")
                    if diagnosis in ("Atelectasis", "Effusion", "Infiltration", "No Finding")
                ]
                if len(detected_diagnosis) != 1:
                    continue
                # if "|" in row["Finding Labels"] or (cooked_folder / row["Image Index"]).exists():
                #     continue  # Skip multilabel
                work_queue.put((raw_folder / row["Image Index"], cooked_folder / row["Image Index"]))
                if monotonic() - run_time > 30:
                    logger.info(
                        "Synchronizing database and file system.Total processed %s (%.2f/sec) images.",
                        row_count,
                        (row_count - sub_total) / 30,
                    )
                    sub_total = row_count
                    work_queue.join()
                    session.commit()
                    logger.info("Sync completed")
                    run_time = monotonic()

                diagnosis_label = detected_diagnosis[0].replace("_", " ")
                if diagnosis_label not in known_diagnosis:
                    logger.info("New diagnosis found: %s", diagnosis_label)
                    diagnosis_instance = Diagnosis(diagnosis=diagnosis_label, ignored=False)
                    session.add(diagnosis_instance)
                    session.flush()
                    session.refresh(diagnosis_instance)
                    known_diagnosis[diagnosis_label] = diagnosis_instance.id

                image = Images(
                    filename=row["Image Index"],
                    patient=int(row["Patient ID"]),
                    age=int(row["Patient Age"]),
                    is_male=row["Patient Gender"].lower() == "m",
                    diagnosis_id=known_diagnosis[diagnosis_label],
                )
                session.add(image)

        logger.info("Parsing finished. Waiting for workers")
        work_queue.join()
        exit_event.set()
        for worker in workers:
            worker.join()


def undersample_dataset() -> None:
    """Undersample dataset.
    The class with the smallest number deleted. After all classes reduced to average number of each class occurrence.
     The average number calculation not take the largest class.
    """
    logger = logging.getLogger("raido.dataset")
    with session_scope() as session:
        logger.info("Removing all ignore marks from images and diagnosis")
        session.query(Images).update({Images.ignored: False})
        session.query(Diagnosis).update({Diagnosis.ignored: False})
        diagnosis_count = {}
        for diagnosis in get_diagnosis_list(session):
            diagnosis_count[diagnosis] = (
                session.query(Images).join(Diagnosis, Images.diagnosis).filter(Diagnosis.diagnosis == diagnosis).count()
            )

    with session_scope() as session:
        # minority_classes = sorted(diagnosis_count, key=diagnosis_count.get)[:9]
        majority_class = sorted(diagnosis_count, key=diagnosis_count.get)[-1]
        # for minority_class in minority_classes:
        #     logger.info(
        #         "Marking diagnosis: '%s' as ignored with total images %s",
        #         minority_class,
        #         diagnosis_count[minority_class],
        #     )
        #     for row in (
        #         session.query(Images).join(Diagnosis, Images.diagnosis).filter(Diagnosis.diagnosis == minority_class)
        #     ):  # type: Images
        #         row.ignored = True
        #     session.query(Diagnosis).filter(Diagnosis.diagnosis == minority_class).update({Diagnosis.ignored: True})

    with session_scope() as session:
        logger.info(
            "Removing majority class: '%s' from statistics calculation with total images %s",
            majority_class,
            diagnosis_count[majority_class],
        )
        del diagnosis_count[majority_class]
        # for minority_class in minority_classes:
        #     del diagnosis_count[minority_class]
        average = np.rint(np.average(list(diagnosis_count.values())))
        logger.info("Diagnosis average %s", average)

        for diagnosis in get_diagnosis_list(session):
            for is_above in [True, False]:
                for count, image in enumerate(
                    session.query(Images)
                    .join(Diagnosis, Images.diagnosis)
                    .filter(Diagnosis.diagnosis == diagnosis)
                    .filter(Images.is_above.is_(is_above))
                    .order_by(func.random())
                ):  # type: Tuple[int, Images]
                    image.ignored = count > (average // 2)
