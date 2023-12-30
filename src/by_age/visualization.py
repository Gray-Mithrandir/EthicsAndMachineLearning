"""Plots dataset analysis with  dataset reduction by patient sex"""
from __future__ import annotations

import logging
from itertools import product
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from matplotlib import use
from sqlalchemy import func

from config import PlotSettings
from by_age.database import (
    session_scope,
    Images,
    Diagnosis,
    TrainHistory,
    RunStatus,
    ClassPerformance,
    ClassificationReport,
)
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from by_age.database import get_diagnosis_list


def plot_dataset_reduction(export_path: Path, train: bool) -> None:
    """Plot validation dataset vs reduction

    Parameters
    ----------
    export_path: Path
        Folder to save plots
    train: bool
        If set plot train dataset, otherwise validation
    """
    logger = logging.getLogger("raido.dataset")
    logger.info("Plotting %s reduction", "train" if train else "validation")
    use("Agg")

    data_rows = []
    logger.info("Scanning dataset")
    with session_scope() as session:
        for reduction in range(0, 100, 1):
            for diagnosis in get_diagnosis_list(session):
                for is_above in [True, False]:
                    count = (
                        session.query(Images)
                        .join(Diagnosis, Images.diagnosis)
                        .filter(Diagnosis.diagnosis == diagnosis)
                        .filter(Images.ignored.is_(False))
                        .filter(Images.is_validation.is_(not train))
                        .filter(Images.is_train.is_(train))
                        .filter(Images.is_above.is_(is_above))
                        .filter(Images.reduction_index <= 100 - reduction)
                        .count()
                    )
                    data_rows.append([count, diagnosis, "Above" if is_above else "Below", reduction])

    logger.info("Creating dataframe ")
    dataframe = pd.DataFrame(data_rows, columns=["Count", "Diagnosis", "Age", "Reduction"])

    grid = sns.FacetGrid(dataframe, col="Diagnosis", hue="Age", col_wrap=2, height=2.5)
    grid.map(plt.plot, "Reduction", "Count")
    grid.fig.tight_layout(w_pad=1)
    grid.add_legend()
    grid.fig.subplots_adjust(top=0.9)
    logger.info("Saving validation dataset plot")
    if train:
        grid.fig.suptitle("Train dataset")
        filename = "train_dataset_vs_reduction.png"
    else:
        grid.fig.suptitle("Validation dataset")
        filename = "validation_dataset_vs_reduction.png"
    grid.fig.savefig(export_path / filename)

    plt.close()
    logger.info("Validation dataset plot - done")


def plot_rebalanced_dataset(export_path: Path) -> None:
    """Plot dataset before and after balancing

    Parameters
    ----------
    export_path: Path
        Folder to save plots
    """
    logger = logging.getLogger("raido.dataset")
    logger.info("Plotting balancing information")
    plot_settings = PlotSettings()
    use("Agg")
    with session_scope() as session:
        for all_images in [True, False]:
            query = session.query(func.count(Images.id))
            if not all_images:
                query = query.filter(Images.ignored.is_(False))
            diagnosis_names = []
            diagnosis_counts = []
            logger.info("Scanning for dataset include ignored %s", all_images)
            for diagnosis_row in session.query(Diagnosis).order_by(Diagnosis.diagnosis.asc()):  # type: Diagnosis
                if (not all_images) and diagnosis_row.ignored:
                    continue
                diagnosis_counts.append(
                    query.join(Diagnosis, Images.diagnosis)
                    .filter(Diagnosis.diagnosis == diagnosis_row.diagnosis)
                    .one()[0]
                )
                diagnosis_names.append(diagnosis_row.diagnosis)

            fig, ax = plt.subplots(
                figsize=plot_settings.plot_size, dpi=plot_settings.plot_dpi, subplot_kw={"aspect": "equal"}
            )
            fig.tight_layout()
            wedges, _, _ = plt.pie(diagnosis_counts, autopct="%1.1f%%")

            ax.legend(
                wedges,
                [
                    f"{diag} - {cnt / (np.sum(diagnosis_counts) / 100.0):.1f}% ({cnt})"
                    for diag, cnt in zip(diagnosis_names, diagnosis_counts)
                ],
                title="Diagnosis",
            )
            if all_images:
                title = "Before balancing"
                file_name = "dataset_before_balance.png"
            else:
                title = "Alter balancing"
                file_name = "dataset_after_balance.png"
            ax.set_title(title)
            plt.savefig(export_path / file_name)
    plt.close()


def plot_labels(export_path: Path) -> None:
    """Plot dataset labels vs age and sex

    Parameters
    ----------
    export_path: Path
        Folder to save plots
    """
    logger = logging.getLogger("raido.dataset")
    logger.info("Plotting dataset class correlation")
    plot_settings = PlotSettings()
    use("Agg")

    data_rows = []
    logger.info("Scanning dataset")
    with session_scope() as session:
        for image_row in session.query(Images).filter(Images.ignored.is_(False)):  # type: Images
            data_rows.append((image_row.diagnosis.diagnosis, "Male" if image_row.is_male else "Female", image_row.age))
    logger.info("Creating dataframe")
    dataframe = pd.DataFrame(data_rows, columns=["Diagnosis", "Sex", "Age"])

    fig = plt.figure(figsize=plot_settings.plot_size, dpi=plot_settings.plot_dpi)
    fig.tight_layout()

    viol_plt = sns.violinplot(
        data=dataframe,
        x="Diagnosis",
        y="Age",
        hue="Sex",
        split=True,
        inner="quart",
        fill=False,
        palette={"Male": "g", "Female": ".35"},
        cut=0
    )
    viol_plt.set(title="Diagnosis distribution")
    plt.xticks(rotation=45)
    logger.info("Saving diagnosis distribution")
    viol_plt.get_figure().savefig(export_path / "diagnosis_distribution_by_age.png")
    plt.close()

    fig = plt.figure(figsize=plot_settings.plot_size, dpi=plot_settings.plot_dpi)
    fig.tight_layout()
    plt.xticks(rotation=45)
    bar_plot = sns.countplot(data=dataframe,
                             x="Diagnosis",
                             hue="Sex")
    bar_plot.set(title="Diagnosis distribution")
    logger.info("Saving diagnosis distribution")
    bar_plot.get_figure().savefig(export_path / "diagnosis_distribution_by_count.png")
    plt.close()
    logger.info("Diagnosis distribution plot - done")


def plot_train(export_path: Path) -> None:
    """Plot train, validation, and test accuracy

    Parameters
    ----------
    export_path: Path
        Folder to save plots
    """
    logger = logging.getLogger("raido.dataset")
    logger.info("Plotting training accuracy")
    plot_settings = PlotSettings()
    use("Agg")

    data_rows = []
    logger.info("Scanning dataset")
    with session_scope() as session:
        for train_record in session.query(TrainHistory):  # type: TrainHistory
            epoch_max = (
                session.query(func.max(TrainHistory.epoch))
                .filter(TrainHistory.network_id == train_record.network_id)
                .one()[0]
            )
            if train_record.epoch != epoch_max:
                continue
            if train_record.network.reduce_by_above is True:
                gender = "Above"
            elif train_record.network.reduce_by_above is False:
                gender = "Below"
            else:
                gender = "Both"
            data_rows.append(
                (
                    train_record.network.network,
                    train_record.network.reduction,
                    gender,
                    train_record.train_accuracy,
                    "Train",
                )
            )
            data_rows.append(
                (
                    train_record.network.network,
                    train_record.network.reduction,
                    gender,
                    train_record.validation_accuracy,
                    "Validation",
                )
            )
            for test_row in (
                    session.query(ClassPerformance)
                            .filter(ClassPerformance.network_id == train_record.network_id)
                            .filter(ClassPerformance.diagnosis_id.is_(None))
            ):
                if test_row.test_metric_is_above is True:
                    test_age = "Test(Above)"
                elif test_row.test_metric_is_above is False:
                    test_age = "Test(Below)"
                else:
                    test_age = "Test"
                data_rows.append(
                    (
                        train_record.network.network,
                        train_record.network.reduction,
                        gender,
                        test_row.accuracy,
                        test_age,
                    )
                )
    dataframe = pd.DataFrame(data_rows, columns=["Network", "Reduction", "Reduce by", "Accuracy", "Dataset"])
    sns.set(rc={"figure.figsize": plot_settings.plot_size})
    grid = sns.FacetGrid(dataframe, col="Reduce by", hue="Dataset", row="Network", height=3.5)
    grid.map(plt.plot, "Reduction", "Accuracy", marker="o")
    grid.fig.tight_layout(w_pad=1)
    grid.add_legend()
    grid.fig.subplots_adjust(top=0.9)
    grid.fig.suptitle("Train performance")
    logger.info("Saving train performance plot")
    grid.fig.savefig(export_path / "train_performance.png")
    plt.close()
    logger.info("Train dataset plot - done")


def plot_class_performance(export_path: Path, network: str) -> None:
    """Plot performance per diagnosis

    Parameters
    ----------
    export_path: Path
        Folder to save plots
    network: str
        Network to analyze
    """
    logger = logging.getLogger("raido.dataset")
    logger.info("Plotting class performance")
    use("Agg")
    plot_settings = PlotSettings()
    sns.set(rc={"figure.figsize": plot_settings.plot_size})

    data_rows = []
    logger.info("Scanning dataset")
    with session_scope() as session:
        for performance_row in (
                session.query(ClassPerformance)
                        .join(Diagnosis, ClassPerformance.diagnosis)
                        .join(RunStatus, ClassPerformance.network)
                        .filter(RunStatus.network == network)
        ):  # type: ClassPerformance
            if performance_row.network.reduce_by_above is True:
                age_metric = "Above"
            elif performance_row.network.reduce_by_above is False:
                age_metric = "Below"
            else:
                age_metric = "Both"
            if performance_row.test_metric_is_above is True:
                test_dataset = "Above"
            elif performance_row.test_metric_is_above is False:
                test_dataset = "Below"
            else:
                test_dataset = "Both"

            data_rows.append(
                (
                    performance_row.diagnosis.diagnosis,
                    performance_row.network.reduction,
                    test_dataset,
                    age_metric,
                    performance_row.accuracy,
                )
            )

        for performance_row in (
                session.query(ClassPerformance)
                        .join(RunStatus, ClassPerformance.network)
                        .filter(RunStatus.network == network)
                        .filter(ClassPerformance.diagnosis_id.is_(None))
        ):  # type: ClassPerformance
            if performance_row.network.reduce_by_above is True:
                age_metric = "Above"
            elif performance_row.network.reduce_by_above is False:
                age_metric = "Below"
            else:
                age_metric = "Both"
            if performance_row.test_metric_is_above is True:
                test_dataset = "Above"
            elif performance_row.test_metric_is_above is False:
                test_dataset = "Below"
            else:
                test_dataset = "Both"

            data_rows.append(
                ("All", performance_row.network.dataset_size, test_dataset, age_metric, performance_row.accuracy)
            )

        dataframe = pd.DataFrame(data_rows, columns=["Diagnosis", "Reduction", "Dataset", "Reduce by", "Accuracy"])

        grid = sns.FacetGrid(dataframe, col="Reduce by", hue="Diagnosis", row="Dataset", height=3.5, palette="tab20")
        grid.map(plt.plot, "Reduction", "Accuracy", marker="o")
        grid.fig.tight_layout(w_pad=1)
        grid.add_legend()
        grid.fig.subplots_adjust(top=0.9)
        grid.fig.suptitle(f"{network} per class performance")
        grid.fig.savefig(export_path / "class_performance.png")

        pair_plot = sns.pairplot(dataframe, hue="Diagnosis", corner=True, palette="tab20")
        pair_plot.fig.subplots_adjust(top=0.9)
        pair_plot.fig.suptitle(f"{network} per class performance")
        pair_plot.fig.savefig(export_path / "class_pair_plot.png")
        plt.close()


def plot_classification_report(export_path: Path, network: str) -> None:
    """Plot performance per diagnosis

    Parameters
    ----------
    export_path: Path
        Folder to save plots
    network: str
        Network to analyze
    """
    logger = logging.getLogger("raido.dataset")
    logger.info("Plotting classification report")
    use("Agg")
    plot_settings = PlotSettings()
    sns.set(rc={"figure.figsize": plot_settings.plot_size})

    data_rows = []
    logger.info("Scanning dataset")
    with session_scope() as session:
        for performance_row in (
                session.query(ClassificationReport)
                        .join(RunStatus, ClassificationReport.network)
                        .filter(RunStatus.network == network)
        ):  # type: ClassificationReport
            if performance_row.network.reduce_by_above is True:
                test_metric = "Above"
            elif performance_row.network.reduce_by_above is False:
                test_metric = "Below"
            else:
                test_metric = "Both"
            if performance_row.test_metric_is_above is True:
                test_dataset = "Above"
            elif performance_row.test_metric_is_above is False:
                test_dataset = "Below"
            else:
                test_dataset = "Both"

            data_rows.append(
                (
                    performance_row.label,
                    performance_row.network.reduction,
                    test_dataset,
                    test_metric,
                    performance_row.precision,
                    performance_row.recall,
                    performance_row.f1_score,
                )
            )

        dataframe = pd.DataFrame(
            data_rows, columns=["Metric", "Reduction", "Dataset", "Reduce by", "Precision", "Recall", "F1 Score"]
        )

        for sub_metric in ["Precision", "Recall", "F1 Score"]:
            grid = sns.FacetGrid(dataframe, col="Reduce by", hue="Metric", row="Dataset", height=3.5, palette="tab20")
            grid.map(plt.plot, "Reduction", sub_metric, marker="o")
            grid.fig.tight_layout(w_pad=1)
            grid.add_legend()
            grid.fig.subplots_adjust(top=0.9)
            grid.fig.suptitle(f"{network} per class performance")
            grid.fig.savefig(export_path / f"classification_by_{sub_metric.lower()}.png")
        plt.close()


def plot_confusion_matrix(matrix: NDArray, export_path: Path, labels: Tuple[str, ...]) -> None:
    """Plot confusion matrix:

    Parameters
    ----------
    matrix: NDArray
        Matrix to plot
    export_path: Path
        File path to save plot
    labels: Tuple[str, ...]
        Diagnosis labels
    """
    use("Agg")
    plot_settings = PlotSettings()
    accuracy = np.trace(matrix) / np.sum(matrix).astype("float")
    misclass = 1 - accuracy

    fig = plt.figure(figsize=plot_settings.plot_size, dpi=plot_settings.plot_dpi)
    fig.tight_layout()
    plt.imshow(matrix, interpolation="nearest", cmap=plt.get_cmap("Blues"))
    plt.title("Confusion matrix")
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    result = matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]

    thresh = result.max() / 1.5
    for i, j in product(range(result.shape[0]), range(result.shape[1])):
        plt.text(
            j,
            i,
            f"{result[i, j]:.1f}",
            horizontalalignment="center",
            color="white" if result[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel(
        f"Predicted label\naccuracy={accuracy:0.4f}; misclass={misclass:0.4f}"
    )
    plt.savefig(export_path / "confusion_matrix.png")
    plt.close()
