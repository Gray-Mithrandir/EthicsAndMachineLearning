"""Dataset manipulation utilities for dataset corruption based on patient sex"""
from __future__ import annotations

import logging
from pathlib import Path
from random import choice
from typing import Union

import numpy as np
import tensorflow as tf
from sqlalchemy import func

from config import PreProcessing
from corruption_by_sex.database import (Diagnosis, Images, get_diagnosis_list,
                                        session_scope)


def create_test_set() -> None:
    """Create test set.
    The test set is same for all network, balance, and corruption index
    """
    pre_processing = PreProcessing()
    logger = logging.getLogger("raido.dataset")
    with session_scope() as session:
        diagnosis_list = get_diagnosis_list(session)
    logger.info("Updating labels for test set")
    for diagnosis in diagnosis_list:
        for is_male in [True, False]:
            with session_scope() as session:
                for row in (
                    session.query(Images)
                    .join(Diagnosis, Images.diagnosis)
                    .filter(Diagnosis.diagnosis == diagnosis)
                    .filter(Images.ignored.is_(False))
                    .filter(Images.is_male.is_(is_male))
                    .order_by(func.random())
                    .limit(pre_processing.test_images // 2)
                ):
                    row.is_test = True


def create_validation_set() -> None:
    pre_processing = PreProcessing()
    logger = logging.getLogger("raido.dataset")
    logger.info("Updating labels for validation set")
    with session_scope() as session:
        for diagnosis in get_diagnosis_list(session):
            diagnosis_count = (
                session.query(Images)
                .join(Diagnosis, Images.diagnosis)
                .filter(Images.ignored.is_(False))
                .filter(Images.is_test.is_(False))
                .filter(Diagnosis.diagnosis == diagnosis)
                .count()
            )

            for row in (
                session.query(Images)
                .join(Diagnosis, Images.diagnosis)
                .filter(Diagnosis.diagnosis == diagnosis)
                .filter(Images.ignored.is_(False))
                .filter(Images.is_test.is_(False))
                .order_by(func.random())
                .limit(diagnosis_count * pre_processing.validation_images_fraction)
            ):  # type: Images
                row.is_validation = True


def create_train_set() -> None:
    """Create a validation set"""
    logger = logging.getLogger("raido.dataset")
    logger.info("Updating train for validation set")
    with session_scope() as session:
        session.query(Images).filter(Images.ignored.is_(False)).filter(Images.is_test.is_(False)).filter(
            Images.is_validation.is_(False)
        ).update({Images.is_train: True})


def create_corrupted_labels() -> None:
    """Create corrupted images labels"""
    diagnosis_ids = []
    with session_scope() as session:
        for diagnosis in session.query(Diagnosis).filter(Diagnosis.ignored.is_(False)):
            if diagnosis.diagnosis == "No Finding":
                continue
            diagnosis_ids.append(diagnosis.id)

    with session_scope() as session:
        for image in (
            session.query(Images)
            .join(Diagnosis, Images.diagnosis)
            .filter(Images.ignored.is_(False))
            .filter(Images.is_test.is_(False))
        ):  # type: Images
            if image.diagnosis.diagnosis != "No Finding":
                image.corrupted_diagnosis_id = choice([_id for _id in diagnosis_ids if _id != image.diagnosis_id])
            else:
                image.corrupted_diagnosis_id = image.diagnosis_id


def create_corruption_index(ignore_male: Union[bool, None]) -> None:
    """Create corruption index

    Parameters
    ----------
    ignore_male: Union[bool, None]
        Ignore patient sex. None don't ignore
    """
    logger = logging.getLogger("raido.dataset")
    logger.info("Clearing corruption index")
    with session_scope() as session:
        session.query(Images).update({Images.corruption_index: 0})

    logger.info("Creating corruption index")
    with session_scope() as session:
        for is_training in [True, False]:
            logger.info("Processing %s", "train" if is_training else "validation")
            for diagnosis in get_diagnosis_list(session):
                query = (
                    session.query(Images.id)
                    .join(Diagnosis, Images.diagnosis)
                    .filter(Images.ignored.is_(False))
                    .filter(Images.is_test.is_(False))
                    .filter(Diagnosis.diagnosis == diagnosis)
                )
                if is_training:
                    query = query.filter(Images.is_train.is_(True))
                else:
                    query = query.filter(Images.is_validation.is_(True))
                if ignore_male is not None:
                    query = query.filter(Images.is_male.is_(not ignore_male))

                all_ids = [_id[0] for _id in query.order_by(func.random())]
                sub_arrays = np.array_split(np.array(all_ids), 100)[::-1]
                for corruption_index, indexes in enumerate(sub_arrays):
                    if len(indexes) == 0:
                        continue
                    python_index = tuple(int(_idx) for _idx in indexes)
                    session.query(Images).filter(Images.id.in_(python_index)).update(
                        {Images.corruption_index: corruption_index}
                    )


def get_dataset(corruption: float, dataset: str, batch_size: int) -> tf.data.Dataset:
    """Create a dataset

    Parameters
    ----------
    corruption: float
        Dataset corruption percentage. Valid is `dataset` is train or validation
    dataset: str
        Dataset type. Must be one of 'train', 'validation', or 'test'
    batch_size: int
        Batch size

    Returns
    -------
    tf.data.Dataset
        Batched and prefetched dataset
    """
    config = PreProcessing()
    with session_scope() as session:
        diagnosis_list = get_diagnosis_list(session)
        output_size = len(diagnosis_list)

    def _generator():
        with session_scope() as _session:
            query = _session.query(Images).filter(Images.ignored.is_(False))
            if dataset == "train":
                query = query.filter(Images.is_train.is_(True))
            elif dataset == "validation":
                query = query.filter(Images.is_validation.is_(True))
            elif dataset == "test":
                query = query.filter(Images.is_test.is_(True))
            else:
                raise ValueError(f"Dataset type can be 'train'/'validation'/'test'. Got {dataset}")
            for image_row in query.order_by(func.random()):  # type: Images
                if image_row.diagnosis.ignored:
                    continue
                output_array = np.zeros(shape=output_size)
                if corruption > image_row.corruption_index and dataset != "test":
                    diag_index = diagnosis_list.index(image_row.corrupted_diagnosis.diagnosis)
                else:
                    diag_index = diagnosis_list.index(image_row.diagnosis.diagnosis)
                output_array[diag_index] = 1.0

                img_array = tf.keras.utils.load_img(
                    f"{Path('data', 'cooked', image_row.filename)}", color_mode="grayscale"
                )
                img_array = np.expand_dims(img_array, axis=-1)
                image = tf.convert_to_tensor(img_array, dtype=tf.float32)
                output = tf.constant(output_array, dtype=tf.float32)
                yield image, output
        return _generator()

    return (
        tf.data.Dataset.from_generator(
            _generator,
            output_signature=(
                tf.TensorSpec(shape=config.input_shape, dtype=tf.float32),
                tf.TensorSpec(shape=output_size, dtype=tf.float32),
            ),
        )
        .batch(batch_size=batch_size, drop_remainder=dataset == "train")
        .prefetch(tf.data.AUTOTUNE)
    )


def get_test_set_by_diagnosis(
    diagnosis: Union[str, None], is_male: Union[bool, None], batch_size: int
) -> tf.data.Dataset:
    """Return test dataset with single diagnosis

    Parameters
    ----------
    diagnosis: str
        Diagnosis to fetch
    batch_size: int
        Batch size
    is_male: bool
        Patient sex

    Returns
    -------
    tf.data.Dataset
        Batched and prefetched dataset
    """
    config = PreProcessing()
    with session_scope() as session:
        diagnosis_list = get_diagnosis_list(session)
        output_size = len(diagnosis_list)

    def _generator():
        with session_scope() as _session:
            query = _session.query(Images).filter(Images.ignored.is_(False)).filter(Images.is_test.is_(True))
            if diagnosis is not None:
                query = query.join(Diagnosis, Images.diagnosis).filter(Diagnosis.diagnosis == diagnosis)
            if is_male is not None:
                query = query.filter(Images.is_male.is_(is_male))

            for image_row in query.order_by(func.random()):  # type: Images
                if image_row.diagnosis.ignored:
                    continue
                output_array = np.zeros(shape=output_size)
                diag_index = diagnosis_list.index(image_row.diagnosis.diagnosis)
                output_array[diag_index] = 1.0

                img_array = tf.keras.utils.load_img(
                    f"{Path('data', 'cooked', image_row.filename)}", color_mode="grayscale"
                )
                img_array = np.expand_dims(img_array, axis=-1)
                image = tf.convert_to_tensor(img_array, dtype=tf.float32)
                output = tf.convert_to_tensor(output_array, dtype=tf.float32)
                yield image, output
        return _generator()

    return (
        tf.data.Dataset.from_generator(
            _generator,
            output_signature=(
                tf.TensorSpec(shape=config.input_shape, dtype=tf.float32),
                tf.TensorSpec(shape=output_size, dtype=tf.float32),
            ),
        )
        .batch(batch_size=batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
