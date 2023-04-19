import argparse
from typing import Tuple

import matplotlib.pyplot as plt

from numpy import std
from common import PhenotypeFramework as pf
from revolve2.core.optimization import DbId
from matplotlib.colors import LinearSegmentedColormap
from ._load_db import load_db


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

        df = load_db(database, db_id)

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
    def plot_novelty_from_db(cls, database: str, db_id: DbId = DbId("optmodular"), *_) -> None:
        """
        Plot fitness as described at the top of this file.

        :param database: Database where the data is stored.
        :param db_id: Id of the evolutionary process to plot.
        """

        df = load_db(database, db_id)

        nvlt = (
            df[["generation_index", "value"]]
            .groupby(by="generation_index")
            .describe()["value"])

        # bxpl_data = [n["value"].values for _, n in nvlt]

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        fig.suptitle("Novelty Score")

        ax.set_xlabel("Generations")
        ax.set_ylabel("Novelty")
        ax.plot(nvlt[["max", "mean", "min"]], label=["Max", "Mean", "Min"])
        # ax.violinplot(bxpl_data, positions=list(range(1,len(bxpl_data)+1)))
        ax.legend()
        plt.show()



    @classmethod
    def plot_novelty(cls,
                     database: str,
                     novelty_test:Tuple[str, float] = ("chybyshev-dist", None),
                     db_id: DbId = DbId("optmodular")) -> None:
        """
        :param database: name of the db
        :param db_id: id of the database
        :param test: test to be performed to get novelty metric,(default = chybyshev_distance) options:
                    'yates-chi-squared', 'chi-squared', 'hellinger-dist',
                    'manhattan-dist', 'euclidian-dist', 'pcc'
        :return:
        """
        df = load_db(database, db_id)

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
        colors = [(0, 0, 0), (1, 0, 0)]
        cm = LinearSegmentedColormap.from_list(
            "Custom", colors, N=100)

        df = load_db(database, db_id)
        genotypes = df["serialized_multineat_genome"].loc[df["generation_index"] == df["generation_index"].max()].tolist()

        def _get_img(body):
            body_arr, core_pos = body.to_grid()
            body_arr = np.asarray(body_arr)
            x, y, z = body_arr.shape
            cx, cy, cz = core_pos

            img_xy, img_xz, img_yz = np.zeros((size, size)), np.zeros((size, size)), np.zeros((size, size))
            img_xy[cx + size//2][cy + size//2] += 1
            img_xz[cx + size//2][cz + size//2] += 1
            img_yz[cy + size//2][cz + size//2] += 1
            for xe in range(x):
                for ye in range(y):
                    for ze in range(z):
                        elem = body_arr[xe][ye][ze]
                        if isinstance(elem, Brick) or isinstance(elem, ActiveHinge):
                            img_xy[xe - cx + size//2][ye - cy + size//2] += 1
                            img_xz[xe - cx + size//2][ze - cz + size//2] += 1
                            img_yz[ye - cy + size//2][ze - cz + size//2] += 1

            return img_xy, img_xz, img_yz

        images = [_get_img(develop_v1(pf.deserialize(genotype))) for genotype in genotypes]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))



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


def main() -> None:
    """Run this file as a command line tool."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "database",
        type=str,
        help="The database to plot.",
    )
    parser.add_argument(
        "db_id",
        type=str,
        help="The id of the ea optimizer to plot.")
    parser.add_argument(
        "plot",
        type=str,
        help="The type of plot (bricks_hinges, novelty_scores)"
    )
    parser.add_argument(
        "test",
        type=str,
        nargs="?",
        help="Type of test for novelty plotting: yates-chi-squared, chi-squared, hellinger-dist,manhattan-dist, euclidian-dist, pcc"
    )

    args = parser.parse_args()

    {"bricks_hinges": EAPlots.plot_bricks_hinges,
     "novelty": EAPlots.plot_novelty,
     "novelty_db": EAPlots.plot_novelty_from_db}[args.plot](args.database, DbId(args.db_id), args.test)



if __name__ == "__main__":
    main()
