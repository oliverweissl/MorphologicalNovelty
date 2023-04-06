import argparse
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd

from numpy import std
from common.phenotype_framework import PhenotypeFramework as pf
from common.phenotype_framework import PhenotypeFramework
from revolve2.core.optimization import DbId
from _load_db import load_db



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

        df["bricks"], df["hinges"] = zip(*df.serialized_multineat_genome.apply(PhenotypeFramework.get_bricks_hinges_amount))

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
            .groupby(by="generation_index"))

        nvlts = nvlt.describe()["value"]
        bxpl_data = [n["value"].values for _, n in nvlt]

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        fig.suptitle("Novelty Score")

        ax.set_xlabel("Generations")
        ax.set_ylabel("Novelty")
        ax.plot(nvlts[["max", "mean", "min"]], label=["Max", "Mean", "Min"])
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
     "novelty": EAPlots.plot_novelty}[args.plot](args.database, DbId(args.db_id), args.test)



if __name__ == "__main__":
    main()
