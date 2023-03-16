import argparse
import matplotlib.pyplot as plt
import pandas as pd

from numpy import std
from phenotype_framework import PhenotypeFramework as pf
from analysis.repositories.genotype_db import GenotypeDB
from phenotype_framework import PhenotypeFramework
from revolve2.core.database import open_database_sqlite
from revolve2.core.optimization import DbId
from revolve2.genotypes.cppnwin.genotype_schema import DbGenotype
from revolve2.core.optimization.ea.generic_ea import (
    DbEAOptimizer,
    DbEAOptimizerGeneration,
    DbEAOptimizerIndividual,
)
from sqlalchemy.future import select


class EAPlots:

    @classmethod
    def _get_db_to_df(cls, database: str, db_id: DbId) -> pd.DataFrame:
        db = open_database_sqlite(database)
        # read the optimizer data into a pandas dataframe
        df = pd.read_sql(
            select(
                DbEAOptimizer,
                DbEAOptimizerGeneration,
                DbEAOptimizerIndividual,
                DbGenotype,
                GenotypeDB

            ).filter(
                (DbEAOptimizer.db_id == db_id.fullname)
                & (DbEAOptimizerGeneration.ea_optimizer_id == DbEAOptimizer.id)
                & (DbEAOptimizerIndividual.ea_optimizer_id == DbEAOptimizer.id)
                & (DbEAOptimizerIndividual.genotype_id == GenotypeDB.id)
                & (GenotypeDB.body_id == DbGenotype.id)
                & (
                        DbEAOptimizerGeneration.individual_id
                        == DbEAOptimizerIndividual.individual_id
                )
            ),
            db,
        )
        return df

    @classmethod
    def _normalize_list(cls, lst: list) -> list:
        vmin = min(lst)
        vmax = max(lst)
        return [(x - vmin) / (vmax - vmin) for x in lst]

    @classmethod
    def plot_blocks_hinges(cls, database: str, db_id: DbId, *_) -> None:
        """
        Plot fitness as described at the top of this file.

        :param database: Database where the data is stored.
        :param db_id: Id of the evolutionary process to plot.
        """

        df = cls._get_db_to_df(database, db_id)

        df["blocks"], df["hinges"] = zip(*df.serialized_multineat_genome.apply(PhenotypeFramework.get_blocks_hinges_amount))

        # calculate max min avg
        hngs = (
            df[["generation_index", "hinges"]]
            .groupby(by="generation_index")
            .describe()["hinges"]
        )

        blcks = (
            df[["generation_index", "blocks"]]
            .groupby(by="generation_index")
            .describe()["blocks"]
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.set_xlabel("Generations")
        ax1.set_ylabel("Amount of Hinges")
        ax1.plot(hngs[["max", "mean", "min"]], label=["Max", "Mean", "Min"])
        ax1.legend()

        ax2.set_xlabel("Generations")
        ax2.set_ylabel("Amount of Blocks")
        ax2.plot(blcks[["max", "mean", "min"]], label=["Max", "Mean", "Min"])
        ax2.legend()

        plt.show()

    @classmethod
    def plot_novelty(cls, database: str, db_id: DbId, *test) -> None:
        """
        :param database: name of the db
        :param db_id: id of the database
        :param test: test to be performed to get novelty metric,(default = chybyshev_distance) options:
                    'yates-chi-squared', 'chi-squared', 'hellinger-dist',
                    'manhattan-dist', 'euclidian-dist', 'pcc'
        :return:
        """
        test = test[0] if test[0] else "chybyshev_distance"

        df = cls._get_db_to_df(database, db_id)

        generation_groups = df[["generation_index", "serialized_multineat_genome"]].groupby(by="generation_index")
        genome_groups = [data["serialized_multineat_genome"].values for _, data in generation_groups]

        data = []
        for generation in genome_groups:
            novelty_scores = pf.get_novelty_population(generation, normalization="clipping", test=test)
            data.append(cls._normalize_list(novelty_scores))

        avg, vmax, vmin, vstd = [], [], [], []
        for sublist in data:
            vmax.append(max(sublist))
            vmin.append(min(sublist))
            avg.append(sum(sublist) / len(sublist))
            vstd.append(std(sublist))

        fig, ax = plt.subplots()
        ax.set_title(f"Novelty Score for: {test}-Test")
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
        help="The type of plot (blocks_hinges, novelty_scores)"
    )
    parser.add_argument(
        "test",
        type=str,
        nargs="?",
        help="Type of test for novelty plotting: yates-chi-squared, chi-squared, hellinger-dist,manhattan-dist, euclidian-dist, pcc"
    )

    args = parser.parse_args()

    {"blocks_hinges": EAPlots.plot_blocks_hinges,
     "novelty": EAPlots.plot_novelty}[args.plot](args.database, DbId(args.db_id), args.test)



if __name__ == "__main__":
    main()
