"""Visualisation utils"""
import re
from csv import DictReader
from itertools import product
from pathlib import Path
from typing import Dict, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import use
from PIL import Image
from scipy.interpolate import Rbf

from config import settings
from database import ClassificationReport, RunStatus, session_scope
from dataset import get_sample_images
from history import EvaluationReport, EvaluationSample, TrainHistory

# pylint: disable=too-many-locals


def plot_images_sample() -> None:
    """Plots sample images from test dataset"""
    use("Agg")
    cols = [f"{diagnosis}".capitalize() for diagnosis in sorted(settings.preprocessing.target_diagnosis)]
    rows = [f"{sex}" for sex in ["Female", "Male"]]
    images = [Image.open(image) for image in get_sample_images()]

    plt.rcParams.update({"font.size": settings.plot.font_size})
    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(settings.preprocessing.target_diagnosis),
        figsize=settings.plot.size,
        dpi=settings.plot.dpi,
        subplot_kw={"aspect": "equal"},
    )

    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, rotation=90, size="large")

    for ax, img in zip(axes.flatten(), images):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(wspace=0.05, hspace=-0.65)
    plt.savefig(settings.folders.reports_folder / "dataset_examples.png")
    plt.close()


def plot_unbalanced_set():
    """Plot unbalanced pie chart"""
    use("Agg")
    diagnosis_counts = {}
    with open(settings.folders.root_folder / "Data_Entry.csv", "r", encoding="utf-8") as csv_fh:
        reader = DictReader(csv_fh)
        for row in reader:
            detected_diagnosis = [
                diagnosis
                for diagnosis in row["Finding Labels"].split("|")
                if diagnosis in settings.preprocessing.target_diagnosis
            ]
            if len(detected_diagnosis) != 1:
                continue  # Ignore multi-labeled images
            diagnosis_counts[detected_diagnosis[0]] = diagnosis_counts.get(detected_diagnosis[0], 0) + 1

    fig, ax = plt.subplots(figsize=settings.plot.size, dpi=settings.plot.dpi, subplot_kw={"aspect": "equal"})
    fig.tight_layout()
    plt.rcParams.update({"font.size": settings.plot.font_size})

    wedges, _, _ = plt.pie(
        [diagnosis_counts[diagnosis] for diagnosis in settings.preprocessing.target_diagnosis], autopct="%1.1f%%"
    )
    ax.legend(
        wedges,
        settings.preprocessing.target_diagnosis,
        title="Diagnosis",
    )

    ax.set_title("Source dataset")
    plt.savefig(settings.folders.reports_folder / "source_dataset.png")
    plt.close()


def plot_labels() -> None:
    """Plot dataset labels vs age and sex"""
    use("Agg")
    data_rows = []
    age_re = re.compile(r"[MF]_(\d{3})_.+_.+")
    for diagnosis in settings.preprocessing.target_diagnosis:
        for image_path in (settings.folders.cooked_folder / diagnosis).rglob("*.png"):
            sex = "Male" if image_path.name.startswith("M") else "Female"
            age = int(age_re.findall(image_path.name)[0])
            data_rows.append((diagnosis, sex, age))

    dataframe = pd.DataFrame(data_rows, columns=["Diagnosis", "Sex", "Age"])

    fig = plt.figure(figsize=settings.plot.size, dpi=settings.plot.dpi)
    fig.tight_layout()
    plt.rcParams.update({"font.size": settings.plot.font_size})

    viol_plt = sns.violinplot(
        data=dataframe,
        x="Diagnosis",
        y="Age",
        hue="Sex",
        split=True,
        inner="quart",
        fill=False,
        palette={"Male": "g", "Female": ".35"},
        cut=0,
    )
    viol_plt.set(title="Diagnosis distribution")
    plt.xticks(rotation=45)
    viol_plt.get_figure().savefig(settings.folders.reports_folder / "diagnosis_distribution_by_age.png")
    plt.close()


def plot_train_loss(train_loss: Tuple[float, ...], val_loss: Tuple[float, ...], export_folder: Path) -> None:
    """Plot train history"""
    fig = plt.figure(figsize=settings.plot.size, dpi=settings.plot.dpi)
    fig.tight_layout()
    plt.rcParams.update({"font.size": settings.plot.font_size})

    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title("Model train loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="lower left")

    plt.savefig(export_folder / "train_loss.png")
    plt.close()


def moving_avg(scalars: Tuple[float, ...], weight: float) -> Tuple[float, ...]:
    """Calculate moving average

    Parameters
    ----------
    scalars: Tuple[float, ...]
        Original values
    weight: float
        Average weight

    Returns
    -------
    Tuple[float, ...]
        Smooth line
    """
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1.0 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return tuple(smoothed)


def plot_train(history: TrainHistory, export_folder: Path) -> None:
    """Plots train history

    Parameters
    ----------
    history: TrainHistory
        History to plot
    export_folder: Path
        Folder to save plots
    """
    use("Agg")
    fig = plt.figure(figsize=settings.plot.size, dpi=settings.plot.dpi)
    fig.tight_layout()
    plt.rcParams.update({"font.size": settings.plot.font_size})

    plt.plot(moving_avg(history.accuracy_history[0], 0.9))
    plt.plot(moving_avg(history.accuracy_history[1], 0.9))
    plt.title("Model train accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")

    plt.savefig(export_folder / "train_accuracy.png")
    plt.close()


def plot_train_per_class(history: TrainHistory, export_folder: Path) -> None:
    """Plots train history

    Parameters
    ----------
    history: TrainHistory
        History to plot
    export_folder: Path
        Folder to save plots
    """
    use("Agg")
    fig = plt.figure(figsize=settings.plot.size, dpi=settings.plot.dpi)
    fig.tight_layout()
    plt.rcParams.update({"font.size": settings.plot.font_size})

    accuracy = {diagnosis: [] for diagnosis in settings.preprocessing.target_diagnosis}
    for train_step, _ in history:
        for diagnosis in settings.preprocessing.target_diagnosis:
            accuracy[diagnosis].append(train_step.precision(diagnosis))

    for diagnosis in settings.preprocessing.target_diagnosis:
        plt.plot(moving_avg(accuracy[diagnosis], 0.9))
    plt.title("Model train accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(settings.preprocessing.target_diagnosis, loc="upper left")

    plt.savefig(export_folder / "class_train_accuracy.png")
    plt.close()


def plot_confusion_matrix(metrics: EvaluationReport, export_folder: Path) -> None:
    """Plot confusion matrix for each evelation reports

    Parameters
    ----------
    metrics: EvaluationReport
        Evaluation report
    export_folder: Path
        Folder to save reports
    """
    use("Agg")
    labels = settings.preprocessing.target_diagnosis

    for group in ["male", "female", "common"]:
        fig = plt.figure(figsize=settings.plot.size, dpi=settings.plot.dpi)
        fig.tight_layout()
        plt.rcParams.update({"font.size": settings.plot.font_size})

        matrix = getattr(metrics, group).confusion_matrix
        accuracy = np.trace(matrix) / np.sum(matrix).astype("float")
        misclass = 1 - accuracy

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
        plt.xlabel(f"Predicted label\naccuracy={accuracy:0.4f}; misclass={misclass:0.4f}")
        plt.savefig(export_folder / f"{group}_confusion_matrix.png")
        plt.close()


def plot_prediction(samples: Dict[str, Dict[str, EvaluationSample]], export_folder: Path) -> None:
    """Plot evaluation sample

    Parameters
    ----------
    samples: Dict[str, Dict[str, EvaluationSample]]
        Evaluation samples as {Male|Female: {Diagnosis: EvaluationSample}}
    export_folder: Path
        Folder to save sample plot
    """
    use("Agg")
    cols = [f"{diagnosis}".capitalize() for diagnosis in sorted(settings.preprocessing.target_diagnosis)]
    rows = [f"{sex}" for sex in ["Female", "Male"]]
    num_rows = 2
    num_cols = len(settings.preprocessing.target_diagnosis)

    fig, axes = plt.subplots(
        nrows=2 * num_rows,
        ncols=num_cols,
        figsize=settings.plot.size,
        dpi=settings.plot.dpi,
        subplot_kw={"aspect": "equal"},
    )
    fig.tight_layout()
    plt.rcParams.update({"font.size": settings.plot.font_size})
    for row_offset, sex in zip([2 * i for i in range(num_rows)], ["Female", "Male"]):
        for col_offset, diagnosis in enumerate(settings.preprocessing.target_diagnosis):
            # Image
            axes[row_offset][col_offset].grid(False)
            axes[row_offset][col_offset].set_xticklabels([])
            axes[row_offset][col_offset].set_yticklabels([])
            axes[row_offset][col_offset].set_xticks([])
            axes[row_offset][col_offset].set_yticks([])
            axes[row_offset][col_offset].imshow(Image.open(samples[sex][diagnosis].image), cmap=mpl.colormaps["binary"])
            # Bars
            axes[row_offset + 1][col_offset].grid(False)
            bar_plot = axes[row_offset + 1][col_offset].bar(
                list(i for i in range(num_cols)),
                samples[sex][diagnosis].probabilities,
                color="#777777",
            )
            for rect in bar_plot:
                height = rect.get_height()
                axes[row_offset + 1][col_offset].text(
                    rect.get_x() + rect.get_width() / 2.0, height, f"{100 * height:.0f}%", ha="center", va="bottom"
                )

            axes[row_offset + 1][col_offset].set_xticklabels([])
            axes[row_offset + 1][col_offset].set_yticklabels([])
            axes[row_offset + 1][col_offset].set_xticks([])
            axes[row_offset + 1][col_offset].set_yticks([])
            axes[row_offset + 1][col_offset].set_ylim([0, 1])
            bar_plot[int(np.argmax(samples[sex][diagnosis].probabilities))].set_color("red")
            bar_plot[settings.preprocessing.target_diagnosis.index(samples[sex][diagnosis].true_label)].set_color(
                "blue"
            )

    for offset, col in enumerate(cols):
        axes[0][offset].set_title(col)

    for offset, row in enumerate(rows):
        axes[2 * offset][0].set_ylabel(row, rotation=90)
    fig.subplots_adjust(wspace=0.05, hspace=-0.65)
    plt.savefig(export_folder / "label_prediction.png")
    plt.close()


def plot_train_accuracy(network: str, reduce_by_male: Union[bool, None], export_folder: Path) -> None:
    """Plot train and validation accuracy

    Parameters
    ----------
    network: str
        Network name to plot
    reduce_by_male: Union[bool, None]
        Train type. Set dataset reduced by male, clear female, none both
    export_folder: Path
        Folder to save plots
    """
    use("Agg")
    fig, axes = plt.subplots(nrows=3, ncols=1, dpi=settings.plot.dpi, subplot_kw={"aspect": "equal"}, sharex=True)
    plt.rcParams.update({"font.size": settings.plot.font_size})
    with session_scope() as session:
        corruption = []
        reduction = []
        train_accuracy = []
        val_accuracy = []
        test_accuracy = []
        for record in (
            session.query(RunStatus, ClassificationReport)
            .join(ClassificationReport)
            .filter(RunStatus.network == network)
            .filter(RunStatus.reduce_by_male.is_(reduce_by_male))
            .filter(ClassificationReport.label == "macro avg")
            .filter(ClassificationReport.test_metric_is_male.is_(None))
        ):  # type: Tuple[RunStatus, ClassificationReport]
            corruption.append(record[0].corruption)
            reduction.append(record[0].reduction)
            train_accuracy.append(record[0].train_accuracy)
            val_accuracy.append(record[0].validation_accuracy)
            test_accuracy.append(record[1].precision)
        xi = np.linspace(min(corruption), max(corruption), np.unique(corruption).size)
        yi = np.linspace(min(reduction), max(reduction), np.unique(reduction).size)
        xi, yi = np.meshgrid(xi, yi)

        for title, ax, metric in zip(
            ("Train", "Validation", "Test"), axes, (train_accuracy, val_accuracy, test_accuracy)
        ):
            # Interpolate
            ax.set_ylabel(title)
            rbf = Rbf(reduction, corruption, metric, function="linear", smooth=4)
            zi = rbf(xi, yi)

            surf = ax.imshow(
                zi,
                vmin=min(metric),
                vmax=max(metric),
                origin="lower",
                extent=[min(reduction), max(reduction), min(corruption), max(corruption)],
                cmap=plt.get_cmap("RdBu"),
            )

        # fig.subplots_adjust(wspace=0.31)
        fig.text(0.62, 0.03, "Dataset reduction", ha="center")
        fig.text(0.35, 0.5, "Label corruption", va="center", rotation="vertical")
        fig.colorbar(surf, ax=axes.ravel().tolist(), location="right")
        plt.savefig(export_folder / "train_accuracy_history.png")
        plt.close()


def plot_class_metrics(network: str, reduce_by_male: Union[bool, None], export_folder: Path) -> None:
    """Plot class metrics

    Parameters
    ----------
    network: str
        Network name to plot
    reduce_by_male: Union[bool, None]
        Train type. Set dataset reduced by male, clear female, none both
    export_folder: Path
        Folder to save plots
    """
    # use("Agg")
    for metric in ["precision", "recall", "f1_score"]:
        fig, axes = plt.subplots(
            nrows=3,
            ncols=len(settings.preprocessing.target_diagnosis),
            # figsize=settings.plot.size,
            dpi=settings.plot.dpi,
            subplot_kw={"aspect": "equal"},
            sharex=True,
            sharey=True,
        )
        plt.rcParams.update({"font.size": int(settings.plot.font_size * 0.8)})
        with session_scope() as session:
            corruption = []
            reduction = []
            network_ids = []
            for record in (
                session.query(RunStatus)
                .filter(RunStatus.network == network)
                .filter(RunStatus.reduce_by_male.is_(reduce_by_male))
            ):  # type: RunStatus
                corruption.append(record.corruption)
                reduction.append(record.reduction)
                network_ids.append(record.id)
            xi = np.linspace(min(corruption), max(corruption), np.unique(corruption).size)
            yi = np.linspace(min(reduction), max(reduction), np.unique(reduction).size)
            xi, yi = np.meshgrid(xi, yi)
        for sex, sub_ax in zip([True, False, None], axes):
            for diagnosis, ax in zip(settings.preprocessing.target_diagnosis, sub_ax):
                with session_scope() as session:
                    value = []
                    for network_id in network_ids:
                        for record in (
                            session.query(ClassificationReport)
                            .filter(ClassificationReport.network_id == network_id)
                            .filter(ClassificationReport.label == diagnosis)
                            .filter(ClassificationReport.test_metric_is_male.is_(sex))
                        ):  # type: ClassificationReport
                            value.append(getattr(record, metric))

                    rbf = Rbf(reduction, corruption, value, function="linear", smooth=3)
                    zi = rbf(xi, yi)

                    surf = ax.imshow(
                        zi,
                        vmin=min(value),
                        vmax=max(value),
                        origin="lower",
                        extent=[min(reduction), max(reduction), min(corruption), max(corruption)],
                        cmap=plt.get_cmap("RdBu"),
                    )

        for offset, col in enumerate(
            [f"{diagnosis}".capitalize() for diagnosis in sorted(settings.preprocessing.target_diagnosis)]
        ):
            axes[0][offset].set_title(col)
        for offset, row in enumerate(["Male", "Female", "All"]):
            axes[offset][0].set_ylabel(row, rotation=90)
        fig.subplots_adjust(wspace=0.1, hspace=-0.7)
        fig.colorbar(surf, ax=axes.ravel().tolist(), location="right", shrink=0.6)
        fig.text(0.5, 0.2, "Dataset reduction", ha="center")
        fig.text(0.01, 0.5, "Label corruption", va="center", rotation="vertical")
        plt.savefig(export_folder / f"class_history_{metric}.png")
        plt.close()
