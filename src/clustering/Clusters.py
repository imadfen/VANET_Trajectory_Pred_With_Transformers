import pandas as pd
import numpy as np
import os
import pickle
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.utils.config_setup import create_dirs

ROOT = os.getcwd()
ROOT_RESOURCES = ROOT + "/resources/clusters/"


class Clusters(object):
    def __init__(self, embeddings, target, padding_masks, config=None):
        # Flatten the sequence_length and features dimensions
        self.embeddings = embeddings
        self.target = target
        self.df_predicted = pd.DataFrame(
            embeddings.reshape(embeddings.shape[0], -1)
        )  # flatten time dimension data - merge last two dimensions
        self.df_target = pd.DataFrame(
            target.reshape(target.shape[0], -1)
        )  # flatten data merge last two dimensions
        self.padding_masks = padding_masks
        self.config = config

        self.data_pooled = np.mean(embeddings, axis=1)
        self.data_pooled_original = (
            np.ma.masked_array(
                target, mask=np.broadcast_to(~padding_masks[:, :, None], target.shape)
            )
            .mean(axis=1)
            .data
        )

    def cluster_data(self):
        pass

    def save_clusters(self, filename, data):
        if self.config is not None and os.path.exists(self.config["output_dir"]):
            if not os.path.exists(self.config["output_dir"] + "/clusters"):
                create_dirs([self.config["output_dir"] + "/clusters"])
            filepath = os.path.join(self.config["output_dir"] + "/clusters", filename)
        else:
            filepath = os.path.join(ROOT_RESOURCES, filename)

        with open(filepath, "wb") as file:
            pickle.dump(data, file)

    def load(self, filename):
        if self.config is not None and os.path.exists(
            self.config["output_dir"] + "/clusters"
        ):
            filepath = os.path.join(self.config["output_dir"] + "/clusters", filename)
        else:
            filepath = os.path.join(ROOT_RESOURCES, filename)

        with open(filepath, "rb") as file:
            data = pickle.load(file)

        return data

    def get_color_palette(self, num_clusters):
        cmap = cm.get_cmap("hsv")
        COLOR_PALETTE = [cmap(i / num_clusters) for i in range(num_clusters)]

        return COLOR_PALETTE

    def plot_clusters(self, clusters_df: pd.DataFrame, remove_noise: bool = True):
        """Plot (PCA-projected) vehicle trajectories coloured by HDBSCAN cluster."""
        from sklearn.decomposition import PCA

        COLOR_PALETTE = self.get_color_palette(np.max(clusters_df) + 2)
        data = self.data_pooled  # (N, emb_dim)

        pca = PCA(n_components=2)
        coords = pca.fit_transform(data)  # (N, 2)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Vehicle embeddings coloured by HDBSCAN cluster")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")

        for cluster in np.unique(clusters_df):
            if remove_noise and cluster == -1:
                continue
            mask = clusters_df == cluster
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=[COLOR_PALETTE[cluster % len(COLOR_PALETTE)]],
                label=f"Cluster {cluster}",
                s=6, alpha=0.7,
            )

        ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        plt.tight_layout()
        plt.show()


class HDBSCANCluster(Clusters):
    def __init__(
        self,
        embeddings,
        target,
        padding_masks,
        min_cluster_size=5,
        min_samples=30,
        config=None,
    ):
        super().__init__(
            embeddings=embeddings,
            target=target,
            padding_masks=padding_masks,
            config=config,
        )

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.clusterer = None
        self.clusters = None

    def cluster_data(self, data: pd.DataFrame):
        self.clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size, min_samples=self.min_samples
        )
        self.clusters = self.clusterer.fit_predict(data)

        print("Cluster assignments:", len(set(self.clusters)))

        return self.clusters

    def get_silhouette_score(
        self, data: pd.DataFrame, clusters: pd.DataFrame, remove_noise: bool = True
    ):
        num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        if remove_noise:
            data = data[clusters != -1]
            clusters = clusters[clusters != -1]
        if num_clusters > 1:
            score = silhouette_score(data, clusters)
            print(f"Silhouette Score: {score}, num_clusters: {num_clusters}")
        else:
            print("Less than 2 clusters detected.")

    def run(
        self,
        original_data: bool = False,
        remove_noise: bool = True,
        save_data: bool = True,
        show_clusters: bool = False,
    ):
        if original_data:
            data = self.data_pooled_original
        else:
            data = self.data_pooled

        self.cluster_data(data)
        self.get_silhouette_score(data, self.clusters, remove_noise)

        if save_data:
            self.save_clusters(
                f'hdbscan_model{"_original" if original_data else ""}.pkl',
                self.clusterer,
            )
            self.save_clusters(
                f'cluster_labels{"_original" if original_data else ""}.pkl',
                self.clusters,
            )
            self.save_clusters("data_embeddings.pkl", self.embeddings)
            self.save_clusters("data_original.pkl", self.target)
            self.save_clusters("data_padding.pkl", self.padding_masks)

        if show_clusters:
            self.plot_clusters(self.clusters, remove_noise)

    def load_clusters(
        self,
        original_data: bool = False,
        remove_noise: bool = True,
    ):
        clusters = self.load(
            f'cluster_labels{"_original" if original_data else ""}.pkl'
        )
        embeddings = self.load("data_embeddings.pkl")
        target = self.load("data_original.pkl")
        padding_masks = self.load("data_padding.pkl")

        if original_data:
            data = (
                np.ma.masked_array(
                    target,
                    mask=np.broadcast_to(~padding_masks[:, :, None], target.shape),
                )
                .mean(axis=1)
                .data
            )
        else:
            data = np.mean(embeddings, axis=1)

        self.get_silhouette_score(data, clusters, remove_noise)

        return clusters, embeddings, target, padding_masks
