"""Setup and running of the optimize modular program."""
import time
import logging


from random import Random

import multineat
from .genotype import random as random_genotype
from .optimizer import Optimizer
from revolve2.core.database import open_async_database_sqlite
from revolve2.core.optimization import DbId


async def main(novelty_search: bool = True, novelty_weight: float = None) -> None:
    """Run the optimization process."""
    # number of initial mutations for body and brain CPPNWIN networks
    NUM_INITIAL_MUTATIONS = 10

    SIMULATION_TIME = 10
    SAMPLING_FREQUENCY = 5
    CONTROL_FREQUENCY = 60

    POPULATION_SIZE = 100
    OFFSPRING_SIZE = 100 # tournament on only new individuals
    NUM_GENERATIONS = 400


    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    logging.info("Starting optimization")

    # random number generator
    rng = Random()
    rng.seed(6)

    # database
    tme = time.strftime("[%H-%M-%S]", time.localtime())
    db_str = f"/media/oliver/DC7282EC7282CB2A/Users/olive/Desktop/thesis/experiments/db_n_{novelty_weight}_{tme}" if novelty_search else f"/media/oliver/DC7282EC7282CB2A/Users/olive/Desktop/thesis/experiments/db_{tme}"
    database = open_async_database_sqlite(db_str, create=True)

    # unique database identifier for optimizer
    db_id = DbId.root("optmodular")

    # multineat innovation databases
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

    initial_population = [
        random_genotype(innov_db_body, innov_db_brain, rng, NUM_INITIAL_MUTATIONS)
        for _ in range(POPULATION_SIZE)
    ]

    maybe_optimizer = await Optimizer.from_database(
        database=database,
        db_id=db_id,
        innov_db_body=innov_db_body,
        innov_db_brain=innov_db_brain,
        rng=rng,
    )
    if maybe_optimizer is not None:
        optimizer = maybe_optimizer
    else:
        optimizer = await Optimizer.new(
            database=database,
            db_id=db_id,
            initial_population=initial_population,
            rng=rng,
            innov_db_body=innov_db_body,
            innov_db_brain=innov_db_brain,
            simulation_time=SIMULATION_TIME,
            sampling_frequency=SAMPLING_FREQUENCY,
            control_frequency=CONTROL_FREQUENCY,
            num_generations=NUM_GENERATIONS,
            offspring_size=OFFSPRING_SIZE,
        )

    logging.info("Starting optimization process..")

    await optimizer.run(novelty_search=novelty_search, novelty_weight=novelty_weight)

    logging.info("Finished optimizing.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
