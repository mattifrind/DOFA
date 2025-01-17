"""
This file contains visualization code for UMAP. See the main() method for usage examples.
Calling this Python file directly allows to load CSMAE features and visualize them, as an example.
@author Felix Kleinsteuber (contact me if you have questions)
"""

import argparse
import pickle
from typing import Tuple
import numpy as np
import umap
import umap.plot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from convert_labels import convert_labels

BEN19_LABELS = {
    "Urban fabric": 0,
    "Industrial or commercial units": 1,
    "Arable land": 2,
    "Permanent crops": 3,
    "Pastures": 4,
    "Complex cultivation patterns": 5,
    "Land principally occupied by agriculture, with significant areas of natural vegetation": 6,
    "Agro-forestry areas": 7,
    "Broad-leaved forest": 8,
    "Coniferous forest": 9,
    "Mixed forest": 10,
    "Natural grassland and sparsely vegetated areas": 11,
    "Moors, heathland and sclerophyllous vegetation": 12,
    "Transitional woodland, shrub": 13,
    "Beaches, dunes, sands": 14,
    "Inland wetlands": 15,
    "Coastal wetlands": 16,
    "Inland waters": 17,
    "Marine waters": 18,
}


class UMAPVisualizer:
    def __init__(self, X, Y):
        assert len(X.shape) == 2 and len(Y.shape) == 2 and X.shape[0] == Y.shape[0]
        self.X = X
        self.Y = Y
        self.mapper = umap.UMAP()
        print("Fitting UMAP...")
        self.embedded = self.mapper.fit_transform(X)
        print("Fitting complete")

    def plot_classwise_ben19(
        self,
        output_file: str = "umap_viz",
        show: bool = False,
        class_order: list = list(range(19)),
    ):
        """Plot each class vs rest in a 4x5 grid.

        Args:
            output_file (str, optional): Output file name, without extension. Will be saved as png and pdf. Defaults to "umap_viz".
            show (bool, optional): Call plt.show(). Defaults to False.
            class_order (list, optional): Order of the classes in the grid. Defaults to list(range(19)).
        """
        assert self.Y.shape[1] == 19
        assert len(class_order) == 19

        fig, axs = plt.subplots(4, 5, squeeze=True)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        fig.set_size_inches(20, 15)
        # fig.tight_layout()
        fig.delaxes(axs[-1, -1])
        axs = axs.flatten()[:19]
        for i, ax in enumerate(axs):
            class_idx = class_order[i]
            class_name = list(BEN19_LABELS.keys())[i]
            umap.plot.points(
                self.mapper,
                labels=self.Y[:, class_idx],
                ax=ax,
                theme="fire",
                show_legend=False,
            )
            ax.set_title(f"{class_name[:21]} vs rest")

        if output_file is not None:
            fig.savefig(f"{output_file}.pdf", bbox_inches="tight")
            fig.savefig(f"{output_file}.png", bbox_inches="tight")
        if show:
            plt.show()

    def plot_in_one_ben19(self, output_file: str = "umap_viz", show: bool = False):
        """Plot all data points into one plot, mixing colors based on their classes.

        Args:
            output_file (str, optional): Output file name, without extension. Will be saved as png and pdf. Defaults to "umap_viz".
            show (bool, optional): Call plt.show(). Defaults to False.
        """
        assert self.Y.shape[1] == 19
        colors = [
            "#F44336",  # red
            "#E91E63",  # pink
            "#9C27B0",  # purple
            "#673AB7",  # deep purple
            "#3F51B5",  # indigo
            "#2196F3",  # blue
            "#03A9F4",  # light blue
            "#00BCD4",  # cyan
            "#009688",  # teal
            "#4CAF50",  # green
            "#8BC34A",  # light green
            "#CDDC39",  # lime
            "#FFEB3B",  # yellow
            "#FFC107",  # amber
            "#FF9800",  # orange
            "#FF5722",  # deep orange
            "#795548",  # brown
            "#9E9E9E",  # grey
            "#607D8B",  # blue grey
        ]

        fig, ax = plt.subplots(figsize=(12, 6))
        handles = []
        for i in range(19):
            class_name = list(BEN19_LABELS.keys())[i]
            points = self.embedded[self.Y[:, i] == 1]
            ax.scatter(points[:, 0], points[:, 1], s=1, alpha=0.2, c=colors[i])
            handles.append(mpatches.Patch(color=colors[i], label=class_name[:20]))
        ax.legend(handles=handles, loc="center left")
        ax.set_xlim(-16, 23)
        ax.axis("off")
        fig.savefig(f"{output_file}.pdf", bbox_inches="tight")
        fig.savefig(f"{output_file}.png", bbox_inches="tight")
        if show:
            plt.show()


def load_dino_features(feature_file: str, label_file: str) -> Tuple[np.ndarray, np.ndarray]:
    features = np.load(feature_file)
    labels = np.load(label_file)
    assert features.shape[-1] == 384
    print(f"Loaded features of shape {features.shape} and labels of shape {labels.shape}")
    features = np.reshape(features, (-1, 384))
    labels = np.reshape(labels, (-1, 19))
    print(f"Reshaped features to {features.shape} and labels to {labels.shape}")
    return features, labels
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=str, help="Numpy file")
    parser.add_argument("labels", type=str, help="Numpy file")
    parser.add_argument("output", type=str, help="Output file name without extension")
    parser.add_argument("--show", action="store_true", help="Show the plot")
    args = parser.parse_args()

    print("Loading features...")
    X, Y_19 = load_dino_features(args.features, args.labels)
    assert Y_19.shape[1] == 19
    assert X.shape[0] == Y_19.shape[0]

    print("Visualizing S1...")
    viz_s1 = UMAPVisualizer(X, Y_19)
    viz_s1.plot_classwise_ben19(output_file=args.output, show=args.show)
    viz_s1.plot_in_one_ben19(output_file=args.output + "_all", show=args.show)


if __name__ == "__main__":
    main()
