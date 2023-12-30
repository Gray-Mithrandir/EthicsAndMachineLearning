"""Runner for network performance analysis under dataset reduction bused on patient age"""
from __future__ import annotations

import logging
from typing import Union, Tuple
from pathlib import Path

from logger import init_logger
from by_age.database import session_scope, get_diagnosis_list, RunStatus, Base, engine

from by_age.utils import record_history, record_evaluation_result, if_run_exist, record_classification_report

from by_age.preprocessing import cook_dataset
from by_age.dataset import create_reduction_index
from by_age.visualization import (
    plot_dataset_reduction,
    plot_train,
    plot_class_performance, plot_classification_report,
    plot_confusion_matrix
)
from networks.base import NetworkInterface

from by_age import dataset, utils
from importlib import import_module


def main_train_loop(network_to_test: Tuple[str, ...], train_patient_age: Union[bool, None], export_path: Path) -> None:
    logger = logging.getLogger("raido")
    if train_patient_age is not None:
        reduction_list = [0, 20, 40, 60, 80, 100]
    else:
        reduction_list = [0, 10, 20, 30, 40, 50, 60]
    for reduction in reduction_list:
        for network_class in network_to_test:
            single_loop(logger, network_class, reduction, train_patient_age, export_path)


def single_loop(logger, network_class, reduction, train_patient_age, export_path: Path):
    model_class = getattr(import_module(f"networks.{network_class}"), "Network")
    if if_run_exist(network=model_class.name(), reduction=reduction, is_above=train_patient_age):
        logger.warning("Already trained on this settings. Skip")
        return
    output_size = utils.get_output_vector_size()
    network = model_class(output_size)  # type: NetworkInterface
    logger.info("Creating network %s", network.name())
    logger.info("Creating dataset with reduction %s", reduction)
    train_ds = dataset.get_dataset(reduction=reduction, dataset="train", batch_size=network.batch_size)
    validation_ds = dataset.get_dataset(reduction=reduction, dataset="validation", batch_size=network.batch_size)
    logger.info("Starting training")
    history = network.train(train_ds=train_ds, validation_ds=validation_ds)
    run_id = record_history(history=history, reduction=reduction, is_above=train_patient_age, network=network.name())
    logger.info("Starting evaluation")
    with session_scope() as session:
        labels = get_diagnosis_list(session)
        diagnosis_list = list(labels) + [None]

    conf_matrix = network.confusion_matrix(eval_ds=dataset.get_dataset(reduction=0, dataset="test", batch_size=1))
    plot_path = export_path / f"{network.name()}_reduction_{reduction}_s_{train_patient_age}"
    plot_path.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(matrix=conf_matrix, export_path=plot_path, labels=labels)

    for is_above in [True, False, None]:
        logger.info("Creating classification report is_above %s", is_above)
        report = network.classification_report(eval_ds=dataset.get_dataset(reduction=0, dataset="test", batch_size=1),
                                               labels=labels)
        record_classification_report(run_id=run_id, is_above=is_above, report=report)
        for diagnosis in diagnosis_list:
            logger.info("Evaluating on is_above: %s, diagnosis: %s", is_above, diagnosis)
            eval_ds = dataset.get_test_set_by_diagnosis(
                diagnosis=diagnosis, is_above=is_above, batch_size=network.batch_size
            )
            eval_result = network.evaluate(eval_ds)
            record_evaluation_result(run_id=run_id, diagnosis=diagnosis, is_above=is_above, result=eval_result)


def main(network_to_test) -> None:
    """Cook dataset main function"""
    Base.metadata.create_all(bind=engine)
    init_logger(Path("logs", "prepare"))
    logger = logging.getLogger("raido")
    logger.info("Preprocessing images")
    cook_dataset()
    logger.info("Creating test dataset")
    dataset.create_test_set()
    logger.info("Creating validation dataset")
    dataset.create_validation_set()
    logger.info("Creating train dataset")
    dataset.create_train_set()

    for reduction_by_age in [None, True, False]:
        logger.info("Creating reduction indexes")
        create_reduction_index(ignore_above=(not reduction_by_age) if reduction_by_age is not None else None)
        logger.info("Plotting dataset statistics")
        export_path = Path("reports", "reduction_age")
        if reduction_by_age is True:
            export_path = export_path / "by_above"
        elif reduction_by_age is False:
            export_path = export_path / "by_below"
        else:
            export_path = export_path / "both"
        export_path.mkdir(parents=True, exist_ok=True)
        logger.info("Plotting dataset vs reduction plots")
        plot_dataset_reduction(export_path, train=False)
        plot_dataset_reduction(export_path, train=True)
        logger.info("Train on reduction by above %s", reduction_by_age)
        main_train_loop(network_to_test=network_to_test, train_patient_age=reduction_by_age, export_path=export_path)

    logger.info("Plot overall performance")
    plot_train(export_path=Path("reports", "reduction_age"))
    for network_class in network_to_test:
        model_class = getattr(import_module(f"networks.{network_class}"), "Network")
        name = model_class.name()
        network_export_path = Path("reports", "reduction_age") / name
        network_export_path.mkdir(exist_ok=True, parents=True)
        plot_class_performance(export_path=network_export_path, network=name)
        plot_classification_report(export_path=network_export_path, network=name)

if __name__ == "__main__":
    main(("lenet5", "alexnet", "resnet", "vit"))
