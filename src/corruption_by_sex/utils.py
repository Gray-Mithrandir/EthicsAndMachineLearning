"""Helpers utils"""
from __future__ import annotations

from typing import Dict, Union

import tensorflow as tf

from corruption_by_sex.database import (
    ClassificationReport,
    ClassPerformance,
    Diagnosis,
    RunStatus,
    TrainHistory,
    get_diagnosis_list,
    session_scope,
)
from networks.base import EvaluationResult


def record_history(
    history: tf.keras.callbacks.History, corruption: int, is_male: Union[bool, None], network: str
) -> int:
    """Store train history in database

    Parameters
    ----------
    history: tf.keras.callbacks.History
        History callback
    corruption: int
        Dataset corruption
    is_male: Union[bool, None]
        Set if corruption done only to male patient, clear only by female, None both
    network: str
        Network name

    Returns
    -------
    int
        Record ID
    """
    with session_scope() as session:
        run_status = RunStatus(network=network, reduce_by_male=is_male, corruption=corruption)
        for epoch, (train_acc, train_loss, val_acc, val_loss, learn_rate) in enumerate(
            zip(
                history.history["accuracy"],
                history.history["loss"],
                history.history["val_accuracy"],
                history.history["val_loss"],
                history.history["lr"],
            ),
            start=1,
        ):
            train_history = TrainHistory(
                epoch=epoch,
                train_accuracy=train_acc,
                train_loss=train_loss,
                validation_accuracy=val_acc,
                validation_loss=val_loss,
                learning_rate=learn_rate,
            )
            run_status.history.append(train_history)

        session.add(run_status)
        session.flush()
        session.refresh(run_status)
        record_id = run_status.id

    return record_id


def record_evaluation_result(
    run_id: int, diagnosis: Union[str, None], is_male: Union[bool, None], result: EvaluationResult
) -> None:
    """Record evaluation result

    Parameters
    ----------
    run_id: int
        Run ID to assign record
    diagnosis: Union[str, None]
        Diagnosis
    is_male: Union[bool, None]
        Patient sex
    result: EvaluationResult
        Network performance
    """

    with session_scope() as session:
        if diagnosis is not None:
            diagnosis_id = session.query(Diagnosis.id).filter(Diagnosis.diagnosis == diagnosis).one()
            diagnosis_id = diagnosis_id[0]
        else:
            diagnosis_id = None

        performance = ClassPerformance(
            network_id=run_id,
            diagnosis_id=diagnosis_id,
            test_metric_is_male=is_male,
            accuracy=result.accuracy,
            loss=result.loss,
        )
        session.add(performance)


def record_classification_report(run_id: int, is_male: Union[bool, None], report: Dict[str, Dict[str, float]]) -> None:
    """Record evaluation result

    Parameters
    ----------
    run_id: int
        Run ID to assign record
    is_male: Union[bool, None]
        Patient sex
    report: Dict[str, Dict[str, float]]
        Classification report
    """
    with session_scope() as session:
        for label, metrics in report.items():
            if not isinstance(metrics, dict):
                continue
            record = ClassificationReport(
                network_id=run_id,
                test_metric_is_male=is_male,
                label=label,
                precision=metrics.get("precision"),
                recall=metrics.get("recall"),
                f1_score=metrics.get("f1-score"),
                support=metrics.get("support"),
            )
            session.add(record)


def if_run_exist(network: str, corruption: int, is_male: bool) -> bool:
    """Return if run already exist"""
    with session_scope() as session:
        stat = (
            session.query(RunStatus)
            .filter(RunStatus.network == network)
            .filter(RunStatus.corruption == corruption)
            .filter(RunStatus.reduce_by_male.is_(is_male))
            .first()
        )
        if stat is None:
            return False
        return True


def get_output_vector_size() -> int:
    """Return output vector size"""
    with session_scope() as session:
        return len(get_diagnosis_list(session=session))
