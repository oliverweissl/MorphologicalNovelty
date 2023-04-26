
from typing import Tuple

import matplotlib.pyplot as plt
import pandas

import pandas as pd
import numpy as np
import seaborn as sns
from numpy import std
from common import PhenotypeFramework as pf
from revolve2.core.optimization import DbId
from revolve2.genotypes.cppnwin.modular_robot.body_genotype_v1 import develop_v1
from revolve2.core.modular_robot import Body, Brick, ActiveHinge
from ._load_db import load_db_novelty
from ._condenser import Condenser


class EAPlots:
    @classmethod
    def _normalize_list(cls, lst: list) -> list:
        vmin = min(lst)
        vmax = max(lst)
        return [(x - vmin) / (vmax - vmin) for x in lst]

    @classmethod
    def plot_bricks_hinges(cls, database: str, db_id: DbId = DbId("optmodular"), *_) -> None:
        """
        Plot fitness as described at the top of this file.

        :param database: Database where the data is stored.
        :param db_id: Id of the evolutionary process to plot.
        """

        df = load_db_novelty(database, db_id)

        df["bricks"], df["hinges"] = zip(*df.serialized_multineat_genome.apply(pf.get_bricks_hinges_amount))

        # calculate max min avg
        hngs = (
            df[["generation_index", "hinges"]]
            .groupby(by="generation_index")
            .describe()["hinges"]
        )

        blcks = (
            df[["generation_index", "bricks"]]
            .groupby(by="generation_index")
            .describe()["bricks"]
        )

        test = database.replace("experiments/database_", "")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(f"{test}")

        ax1.set_xlabel("Generations")
        ax1.set_ylabel("Amount of Hinges")
        ax1.plot(hngs[["max", "mean", "min"]], label=["Max", "Mean", "Min"])
        ax1.legend()

        ax2.set_xlabel("Generations")
        ax2.set_ylabel("Amount of bricks")
        ax2.plot(blcks[["max", "mean", "min"]], label=["Max", "Mean", "Min"])
        ax2.legend()

        plt.show()

    @classmethod
    def plot_novelty_from_db(cls, database: str, db_id: DbId = DbId("optmodular"), *_) -> float:
        """
        Plot fitness as described at the top of this file.

        :param database: Database where the data is stored.
        :param db_id: Id of the evolutionary process to plot.
        """

        df = load_db_novelty(database, db_id)
        #print(df)

        nvlt = (
            df[["generation_index", "value"]]
            .groupby(by="generation_index")
            .describe()["value"])

        # bxpl_data = [n["value"].values for _, n in nvlt]

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        fig.suptitle(f"Novelty Score {database}")

        ax.set_xlabel("Generations")
        ax.set_ylabel("Novelty")
        ax.plot(nvlt[["max", "mean", "min"]], label=["Max", "Mean", "Min"])
        ax.set_ylim([-0.05, 1.05])
        # ax.violinplot(bxpl_data, positions=list(range(1,len(bxpl_data)+1)))
        ax.legend()
        plt.show()
        return nvlt["mean"].mean()

    @classmethod
    def plot_novelty(cls,
                     database: str,
                     novelty_test: Tuple[str, float] = ("chybyshev-dist", None),
                     db_id: DbId = DbId("optmodular")) -> None:
        """
        :param database: name of the db
        :param db_id: id of the database
        :param test: test to be performed to get novelty metric,(default = chybyshev_distance) options:
                    'yates-chi-squared', 'chi-squared', 'hellinger-dist',
                    'manhattan-dist', 'euclidian-dist', 'pcc'
        :return:
        """
        df = load_db_novelty(database, db_id)

        generation_groups = df[["generation_index", "serialized_multineat_genome"]].groupby(by="generation_index")
        genome_groups = [data["serialized_multineat_genome"].values for _, data in generation_groups]

        data = []
        for generation in genome_groups:
            novelty_scores = pf.get_novelty_population(generation, normalization="clipping", novelty_test=novelty_test)
            data.append(novelty_scores)

        avg, vmax, vmin, vstd = [], [], [], []
        for sublist in data:
            vmax.append(max(sublist))
            vmin.append(min(sublist))
            avg.append(sum(sublist) / len(sublist))
            vstd.append(std(sublist))

        fig, ax = plt.subplots()
        ax.set_title(f"Novelty Score for: {novelty_test}-Test")
        ax.plot(avg, label="average", color="blue")
        ax.boxplot(data, positions=list(range(len(data))))

        ax.plot(vmin, label="min")
        ax.plot(vmax, label="max")
        ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
        print(f"Avg STD: {sum(vstd) / len(vstd)}")
        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_avg_shape(cls, database: str, db_id: DbId = DbId("optmodular"), size: int = 40) -> None:
        assert int(size/2) == size//2, "ERROR: size must be a divisible by 2"
        df = load_db_novelty(database, db_id)
        genotypes = df["serialized_multineat_genome"].loc[df["generation_index"] == df["generation_index"].max()].tolist()



        images = [cls._get_img(develop_v1(pf.deserialize(genotype)), size) for genotype in genotypes]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(database)

        im_xy, im_xz, im_yz = None, None, None
        for img_xy, img_xz, img_yz in images:
            im_xy = im_xy + img_xy if im_xy is not None else img_xy
            im_xz = im_xz + img_xz if im_xz is not None else img_xz
            im_yz = im_yz + img_yz if im_yz is not None else img_yz

        cmap = "Greys"
        ax1.imshow(im_xy, origin='lower', cmap=cmap)
        ax2.imshow(im_xz, origin='lower', cmap=cmap)
        ax3.imshow(im_yz, origin='lower', cmap=cmap)


        ax1.set_title("Average Bodies on the XY-Plane")
        ax2.set_title("Average Bodies on the XZ-Plane")
        ax3.set_title("Average Bodies on the ZY-Plane")

        for ax in [ax1,ax2,ax3]:
            ax.set_yticklabels([])
            ax.set_xticklabels([])

    @classmethod
    def plot_novelty_fintess_averages(cls, n, f):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        sns.boxplot(ax=ax1, data=n.iloc[0], color="blue")
        ax1.set_xticklabels(n.columns)
        ax1.set_ylabel("Novelty Average")
        ax1.set_xlabel("Novelty Configuration")
        ax1.set(ylim=(0, 1))

        sns.boxplot(ax=ax2, data=f.iloc[0], color="red")
        ax2.set_xticklabels(n.columns)
        ax2.set_ylabel("Fitness Average")
        ax2.set_xlabel("Novelty Configuration")
        ax1.set(ylim=(0, None))

        plt.show()

    @classmethod
    def plot_metrics_config(cls, metric: dict, ylabel: str, ymax:float = 1.0):
        fig, axs = plt.subplots(2, 3, figsize=(15, 7))
        fig.supxlabel("Generations")
        fig.suptitle(f"{ylabel} Development of:")
        fig.supylabel(ylabel)
        for i, key in enumerate(metric.keys()):
            tmp_df = pd.DataFrame(metric[key]).T.describe().T
            mx, mn, av, pc = tmp_df["max"].max(), tmp_df["min"].min(), tmp_df["mean"].mean(), tmp_df["mean"].pct_change().mean()

            if ylabel == "Fitness":
                txt = f"Max: {mx:.4f}"
                to_plot = tmp_df["mean"]
            else:
                to_plot = tmp_df[["mean", "min"]]
                txt = f"Min: {mn:.4f}"


            axs[i // 3, i % 3].plot(tmp_df.index, to_plot)
            axs[i // 3, i % 3].fill_between(tmp_df.index, tmp_df["mean"]-tmp_df["std"],tmp_df["mean"]+tmp_df["std"], alpha=0.5,color="grey")

            axs[i // 3, i % 3].plot([], [], ' ', label=txt)
            axs[i // 3, i % 3].plot([], [], ' ', label=f"Average: {av:.4f}")
            axs[i // 3, i % 3].plot([], [], ' ', label=f"Avg. Change: {pc:.2%}")

            axs[i // 3, i % 3].set_ylim([-0.05, ymax])
            axs[i // 3, i % 3].set_xlim([-5, 405])
            axs[i // 3, i % 3].set_title(f"Novelty Config: {key}")
            axs[i // 3, i % 3].legend()

        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_finess_over_novelty(cls, fitness:dict, novelty:dict):
        fig, axs = plt.subplots(2, 3, figsize=(15, 7))
        fig.supxlabel("Generations")
        fig.suptitle(f"Fitness * Novelty for:")
        fig.supylabel("Fitness * Novelty")
        for i, key in enumerate(fitness.keys()):
            tmp_f_df = pd.DataFrame(fitness[key]).T.describe().T
            tmp_n_df = pd.DataFrame(novelty[key]).T.describe().T

            to_plot = tmp_f_df["mean"] * tmp_n_df["mean"]
            av = to_plot.mean()

            axs[i // 3, i % 3].plot(tmp_f_df.index, to_plot)
            axs[i // 3, i % 3].plot([], [], ' ', label=f"Average: {av:.4f}")
            axs[i // 3, i % 3].set_title(f"Novelty Config: {key}")
            axs[i // 3, i % 3].legend()

    @classmethod
    def _get_img(cls, body, size):
        body_arr, core_pos = body.to_grid()
        body_arr = np.asarray(body_arr)
        x, y, z = body_arr.shape
        cx, cy, cz = core_pos

        img_xy, img_xz, img_yz = np.zeros((size, size)), np.zeros((size, size)), np.zeros((size, size))
        img_xy[cx + size // 2][cy + size // 2] += 1
        img_xz[cx + size // 2][cz + size // 2] += 1
        img_yz[cy + size // 2][cz + size // 2] += 1
        for xe in range(x):
            for ye in range(y):
                for ze in range(z):
                    elem = body_arr[xe][ye][ze]
                    if isinstance(elem, Brick) or isinstance(elem, ActiveHinge):
                        img_xy[xe - cx + size // 2][ye - cy + size // 2] += 1
                        img_xz[xe - cx + size // 2][ze - cz + size // 2] += 1
                        img_yz[ye - cy + size // 2][ze - cz + size // 2] += 1

        return img_xy, img_xz, img_yz


