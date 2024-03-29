"""Main runner
Test network module should be passed using command line. See --help
"""
import argparse
import importlib
import logging
from pathlib import Path
from typing import Union

import torch

import database
import visualization
from config import settings
from dataset import create_dataset
from logger import init_logger
from networks.base import NetworkInterface


def plot_image_samples() -> None:
    """Plot dataset information in images samples"""
    logger = logging.getLogger("raido")
    logger.info("Examples images plot")
    visualization.plot_images_sample()
    logger.info("Dataset plot balance")
    visualization.plot_unbalanced_set()
    logger.info("Dataset labels")
    visualization.plot_labels()


def main(network: str, corrupt_by_male: Union[bool, None]) -> None:  # pylint: disable=too-many-locals
    """Evaluate a single network with specific data corruption

    Parameters
    ----------
    network: str
        Network to evaluate
    corrupt_by_male: Union[bool, None]
        Corrupt labels options, set corrupt male, clear female, none both
    """
    init_logger(Path("data", "logs"))
    logger = logging.getLogger("raido")
    logger.info("Starting network %s performance evaluation by corrupting labels %s", network, corrupt_by_male)
    logger.info("Initializing CUDA device")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    logger.info("Loading network class")
    network_module = importlib.import_module(f"networks.{network}")
    network_class = getattr(network_module, "Network")
    for corruption in range(0, settings.run.corruption.end, settings.run.corruption.step):
        logger.info("Starting with corruption - %s", corruption)
        for reduction in range(0, settings.run.reduction.end, settings.run.reduction.step):
            run_id = database.create_new_run(
                network=network_class.name(),
                reduction=reduction,
                corruption=corruption,
                reduce_by_male=corrupt_by_male,
            )
            if run_id is None:
                logger.info("Network already evaluated on this set. Skip")
                continue
            export_folder = Path(
                settings.folders.reports_folder,
                network_class.name(),
                {True: "Male", False: "Female", None: "Both"}[corrupt_by_male],
                f"reduction_{reduction:03d}_corruption_{corruption:03d}",
            )
            export_folder.mkdir(parents=True, exist_ok=True)
            logger.info("Starting with dataset reduction %s", reduction)
            create_dataset(corruption=corruption / 100.0, reduction=reduction / 100.0, only_male=corrupt_by_male)

            if corruption == 0 and reduction == 0:
                plot_image_samples()
            logger.info("Creating network")
            network_instance = network_class()  # type: NetworkInterface
            history = network_instance.train(device=device)
            database.update_train_values(run_id=run_id, history=history)
            visualization.plot_train(history=history, export_folder=export_folder)
            visualization.plot_train_per_class(history=history, export_folder=export_folder)
            evaluation = network_instance.evaluate(device=device)
            database.update_evaluation(run_id=run_id, evaluation=evaluation)
            visualization.plot_confusion_matrix(metrics=evaluation, export_folder=export_folder)
            prediction = network_instance.evaluation_sample(device=device)
            visualization.plot_prediction(samples=prediction, export_folder=export_folder)
            database.mark_evaluation_completed(run_id=run_id)
    visualization.plot_train_accuracy(
        network=network_class.name(),
        reduce_by_male=corrupt_by_male,
        export_folder=settings.folders.reports_folder / network_class.name(),
    )
    visualization.plot_class_metrics(
        network=network_class.name(),
        reduce_by_male=corrupt_by_male,
        export_folder=settings.folders.reports_folder / network_class.name(),
    )


if __name__ == "__main__":
    modules = [file.stem for file in Path("src", "networks").glob("*.py") if file.stem not in ["base", "__init__"]]
    parser = argparse.ArgumentParser(
        prog="Raido",
        description="Ethics, error propagation, and dataset-size effects in artificial neural networks",
        epilog="“Ethics is the compass that guides artificial intelligence towards responsible and beneficial outcomes."
        "\nWithout ethical considerations, AI becomes a tool of chaos and harm.” ― Sri Amit Ray",
    )
    parser.add_argument("-n", "--network", help="Network module to test", choices=modules, required=True)
    parser.add_argument(
        "-s",
        "--target_sex",
        help="Target patient sex for label corruption and reduction",
        choices=["female", "male", "both"],
        required=True,
    )
    args = parser.parse_args()
    if args.target_sex == "female":
        TARGET_SEX = False
    elif args.target_sex == "male":
        TARGET_SEX = True
    else:
        TARGET_SEX = None
    main(network=args.network, corrupt_by_male=TARGET_SEX)
