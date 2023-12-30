"""Runner for network performance analysis under dataset corruption bused on patient sex"""
from __future__ import annotations

import logging
from typing import Union, Tuple
from pathlib import Path

from logger import init_logger
from corruption_by_sex.database import session_scope, get_diagnosis_list, Base, engine

from corruption_by_sex.utils import record_history, record_evaluation_result, if_run_exist, record_classification_report

from corruption_by_sex.preprocessing import cook_dataset
from corruption_by_sex.dataset import create_corruption_index
from corruption_by_sex.visualization import (
    plot_dataset_corruption,
    plot_train,
    plot_class_performance, plot_classification_report,
    plot_confusion_matrix, plot_corrupted_labels, plot_corrupted_dataset
)
from networks.base import NetworkInterface

from corruption_by_sex import dataset, utils
from importlib import import_module


def main_train_loop(network_to_test: Tuple[str, ...], train_patient_sex: Union[bool, None], export_path: Path) -> None:
    logger = logging.getLogger("raido")
    if train_patient_sex is not None:
        corruption_list = [0, 5, 10, 15, 20, 25, 30, 40, 60, 80, 100]
    else:
        corruption_list = [0, 2, 5, 7, 10, 12, 15, 20, 30, 40, 50, 60]
    for corruption in corruption_list:
        for network_class in network_to_test:
            single_loop(logger, network_class, corruption, train_patient_sex, export_path)


def single_loop(logger, network_class, corruption, train_patient_sex, export_path: Path):
    model_class = getattr(import_module(f"networks.{network_class}"), "Network")
    if if_run_exist(network=model_class.name(), corruption=corruption, is_male=train_patient_sex):
        logger.warning("Already trained on this settings. Skip")
        return
    output_size = utils.get_output_vector_size()
    network = model_class(output_size)  # type: NetworkInterface
    logger.info("Creating network %s", network.name())
    logger.info("Creating dataset with corruption %s", corruption)
    train_ds = dataset.get_dataset(corruption=corruption, dataset="train", batch_size=network.batch_size)
    validation_ds = dataset.get_dataset(corruption=corruption, dataset="validation", batch_size=network.batch_size)
    logger.info("Starting training")
    history = network.train(train_ds=train_ds, validation_ds=validation_ds)
    run_id = record_history(history=history, corruption=corruption, is_male=train_patient_sex, network=network.name())
    logger.info("Starting evaluation")
    with session_scope() as session:
        labels = get_diagnosis_list(session)
        diagnosis_list = list(labels) + [None]

    conf_matrix = network.confusion_matrix(eval_ds=dataset.get_dataset(corruption=0, dataset="test", batch_size=1))
    plot_path = export_path / f"{network.name()}_corruption_{corruption}_s_{train_patient_sex}"
    plot_path.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(matrix=conf_matrix, export_path=plot_path, labels=labels)

    for is_male in [True, False, None]:
        logger.info("Creating classification report is_male %s", is_male)
        report = network.classification_report(eval_ds=dataset.get_dataset(corruption=0, dataset="test", batch_size=1),
                                               labels=labels)
        record_classification_report(run_id=run_id, is_male=is_male, report=report)
        for diagnosis in diagnosis_list:
            logger.info("Evaluating on is_male: %s, diagnosis: %s", is_male, diagnosis)
            eval_ds = dataset.get_test_set_by_diagnosis(
                diagnosis=diagnosis, is_male=is_male, batch_size=network.batch_size
            )
            eval_result = network.evaluate(eval_ds)
            record_evaluation_result(run_id=run_id, diagnosis=diagnosis, is_male=is_male, result=eval_result)


def main(network_to_test) -> None:
    """Cook dataset main function"""
    init_logger(Path("logs", "prepare"))
    logger = logging.getLogger("raido")
    Base.metadata.create_all(bind=engine)
    logger.info("Preprocessing images")
    cook_dataset()
    logger.info("Creating test dataset")
    dataset.create_test_set()
    logger.info("Creating validation dataset")
    dataset.create_validation_set()
    logger.info("Creating train dataset")
    dataset.create_train_set()
    logger.info("Plotting corrupted dataset")
    plot_corrupted_labels(export_path=Path("reports", "dataset"))
    plot_corrupted_dataset(export_path=Path("reports", "dataset"))

    for corruption_by_male in [None, True, False]:
        logger.info("Creating corruption indexes")
        create_corruption_index(ignore_male=(not corruption_by_male) if corruption_by_male is not None else None)
        logger.info("Plotting dataset statistics")
        export_path = Path("reports", "corruption_sex")
        if corruption_by_male is True:
            export_path = export_path / "by_male"
        elif corruption_by_male is False:
            export_path = export_path / "by_female"
        else:
            export_path = export_path / "both"
        export_path.mkdir(parents=True, exist_ok=True)
        logger.info("Plotting dataset vs corruption plots")
        plot_dataset_corruption(export_path, train=False)
        plot_dataset_corruption(export_path, train=True)
        logger.info("Train on corruption by male %s", corruption_by_male)
        main_train_loop(network_to_test=network_to_test, train_patient_sex=corruption_by_male, export_path=export_path)

    logger.info("Plot overall performance")
    plot_train(export_path=Path("reports", "corruption_sex"))
    for network_class in network_to_test:
        model_class = getattr(import_module(f"networks.{network_class}"), "Network")
        name = model_class.name()
        network_export_path = Path("reports", "corruption_sex") / name
        network_export_path.mkdir(exist_ok=True, parents=True)
        plot_class_performance(export_path=network_export_path, network=name)
        plot_classification_report(export_path=network_export_path, network=name)

if __name__ == "__main__":
    main(("lenet5", "alexnet", "resnet", "vit"))
