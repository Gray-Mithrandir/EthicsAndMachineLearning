from csv import DictReader
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import use

from config import settings
from pathlib import Path
import re
from PIL import Image
import seaborn as sns


def plot_images_sample():
    use("Agg")
    cols = [f"{diagnosis}".capitalize() for diagnosis in sorted(settings.preprocessing.target_diagnosis)]
    rows = [f"{sex}" for sex in ["Female", "Male"]]
    images = []
    for row_sex in ["Female", "Male"]:
        for col_diagnosis in sorted(settings.preprocessing.target_diagnosis):
            img_path = list(Path(settings.folders.test_folder, "Uni", f"{col_diagnosis}").glob("*.png"))
            img_name = sorted([img.name for img in img_path])
            re_filter = re.compile(f"{row_sex[0]}" + r"_0(1[8-9]|[2-9]{2}).+\.png")  # Skip small images
            selected_img = list(filter(re_filter.match, img_name))[0]
            images.append(Image.open(Path(settings.folders.test_folder, "Uni", f"{col_diagnosis}", selected_img)))

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


def plot_train_accuracy(train_accuracy: Tuple[float, ...], val_accuracy: Tuple[float, ...],
                        export_folder: Path) -> None:
    """Plot train history"""
    fig = plt.figure(figsize=settings.plot.size, dpi=settings.plot.dpi)
    fig.tight_layout()
    plt.rcParams.update({"font.size": settings.plot.font_size})

    plt.plot(train_accuracy)
    plt.plot(val_accuracy)
    plt.title('Model train accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig(export_folder / "train_accuracy.png")
    plt.close()


def plot_train_loss(train_loss: Tuple[float, ...], val_loss: Tuple[float, ...], export_folder: Path) -> None:
    """Plot train history"""
    fig = plt.figure(figsize=settings.plot.size, dpi=settings.plot.dpi)
    fig.tight_layout()
    plt.rcParams.update({"font.size": settings.plot.font_size})

    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('Model train loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower left')

    plt.savefig(export_folder / "train_loss.png")
    plt.close()


def plot_probabilities(male_prob, female_prob):
    """Plot probabilities"""
    data_array = [(prob, "Male") for prob in male_prob] + [(prob, "Female") for prob in female_prob]
    df = pd.DataFrame(data_array, columns=["Probability", "Sex"])
    prob_plot = sns.histplot(df, x="Probability", hue="Sex", stat="count", binrange=(0, 1), multiple="dodge")
    # prob_plot = sns.violinplot(df, x="Probability", y="Sex")
    prob_plot.set(title="Diagnosis distribution")
    plt.show()

def plot_probabilities_violin(male_prob, female_prob):
    """Plot probabilities"""
    data_array = []
    for diag in settings.preprocessing.target_diagnosis:
        for prob in male_prob[diag]:
            data_array.append((prob, "Male", diag))
        for prob in female_prob[diag]:
            data_array.append((prob, "Female", diag))

    df = pd.DataFrame(data_array, columns=["Probability", "Sex", "Diagnosis"])
    prob_plot = sns.violinplot(df, x="Diagnosis", y="Probability", hue="Sex",
                               split=True,
                               inner="quart",
                               fill=False,
                               palette={"Male": "g", "Female": ".35"},
                               cut=0,
                               )
    prob_plot.set(title="Diagnosis distribution")
    plt.show()
