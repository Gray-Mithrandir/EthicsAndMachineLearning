"""Dataset manipulation utilities for dataset reduction based on patient sex"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

from config import PreProcessing
from by_sex.database import session_scope, Images, Diagnosis, get_diagnosis_list
from sqlalchemy import func
import numpy as np
import tensorflow as tf


def create_test_set() -> None:
    """Create test set.
    The test set is same for all network, balance, and reduction index
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


def create_reduction_index(ignore_male: Union[bool, None]) -> None:
    """Create reduction index

    Parameters
    ----------
    ignore_male: Union[bool, None]
        Ignore patient sex. None don't ignore
    """
    logger = logging.getLogger("raido.dataset")
    logger.info("Clearing reduction index")
    with session_scope() as session:
        session.query(Images).update({Images.reduction_index: 0})

    logger.info("Creating reduction index")
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
                for reduction_index, indexes in enumerate(sub_arrays):
                    if len(indexes) == 0:
                        continue
                    python_index = tuple(int(_idx) for _idx in indexes)
                    session.query(Images).filter(Images.id.in_(python_index)).update(
                        {Images.reduction_index: reduction_index}
                    )


def get_dataset(reduction: float, dataset: str, batch_size: int) -> tf.data.Dataset:
    """Create a dataset

    Parameters
    ----------
    reduction: float
        Dataset reduction percentage. Valid is `dataset` is train or validation
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
                query = query.filter(Images.is_train.is_(True)).filter(Images.reduction_index <= 100 - reduction)
            elif dataset == "validation":
                query = query.filter(Images.is_validation.is_(True)).filter(Images.reduction_index <= 100 - reduction)
            elif dataset == "test":
                query = query.filter(Images.is_test.is_(True))
            else:
                raise ValueError(f"Dataset type can be 'train'/'validation'/'test'. Got {dataset}")
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


def remove_reduction_indexes(is_male):
    with session_scope() as session:
        for row in (
                session.query(Images)
                        .join(Diagnosis, Images.diagnosis)
                        .filter(Images.ignored.is_(False))
                        .filter(Images.is_test.is_(False))
        ):  # type: Images
            if row.reduction_index is not None:
                if (row.is_male is not is_male) or (row.reduction_index < 0):
                    row.reduction_index *= -1
