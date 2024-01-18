from pathlib import Path

from config import settings
from networks import lenet5, alexnet, vgg16, vit, resnet
from logger import init_logger
import torch
import visualization


def main():
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda")

    init_logger(Path("data"))
    net = vit.Network()
    # create_dataset(corruption=0.0, reduction=0.0, only_male=None)
    # print("Examples images plot")
    # visualization.plot_images_sample()
    # print("Dataset plot balance")
    # visualization.plot_unbalanced_set()
    # print("Dataset labels")
    # visualization.plot_labels()
    history = net.train(device=device)
    net.evaluate(device=device)
    # visualization.plot_train_accuracy(train_accuracy=tuple(h.train_accuracy for h in history),
    #                                   val_accuracy=tuple(h.val_accuracy for h in history),
    #                                   export_folder=settings.folders.reports_folder)
    # visualization.plot_train_loss(train_loss=tuple(h.train_loss for h in history),
    #                               val_loss=tuple(h.val_loss for h in history),
    #                               export_folder=settings.folders.reports_folder)
    # for only_male in [None, ]:
    #     eval_stat = net.evaluate(device, settings.folders.root_folder / "checkpoint", only_male)
    #     print(f"Only male {only_male}")
    #     print(f"Overall Recall {eval_stat.recall()} Precision {eval_stat.precision()} F1 score {eval_stat.f1_score()}")
    #     for diag in settings.preprocessing.target_diagnosis:
    #         print(f"{diag} Recall {eval_stat.recall(diag)}")
    #     print(f"Confusion matrix: {eval_stat.confusion_matrix}")


if __name__ == '__main__':
    main()
