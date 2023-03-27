from revolve2.core.optimization import DbId
import pandas as pd
from revolve2.core.database import open_database_sqlite
from repositories.ea_database import  (
    DbEAOptimizer,
    DbEAOptimizerGeneration,
    DbEAOptimizerIndividual,
)
from revolve2.genotypes.cppnwin.genotype_schema import DbGenotype
from repositories.genotype_db_schema import GenotypeDB
from repositories.float_schema import FloatDB
from sqlalchemy.future import select

def load_db(database: str, db_id: DbId = DbId("optmodular")) -> pd.DataFrame:
    # open the database
    db = open_database_sqlite(database)
    # read the optimizer data into a pandas dataframe

    df = pd.read_sql(
        select(
            DbEAOptimizer,
            DbEAOptimizerGeneration,
            DbEAOptimizerIndividual,
            DbGenotype,
            GenotypeDB,
            FloatDB

        ).filter(
            (DbEAOptimizer.db_id == db_id.fullname)
            & (DbEAOptimizerGeneration.ea_optimizer_id == DbEAOptimizer.id)
            & (DbEAOptimizerIndividual.ea_optimizer_id == DbEAOptimizer.id)
            & (DbEAOptimizerIndividual.genotype_id == GenotypeDB.id)
            & (FloatDB.id  == DbEAOptimizerIndividual.novelty_id)
            & (GenotypeDB.body_id == DbGenotype.id)
            & (
                    DbEAOptimizerGeneration.individual_id
                    == DbEAOptimizerIndividual.individual_id
            )
        ),
        db,
    )
    return df