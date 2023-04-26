from glob import glob
import pandas as pd
from tqdm import tqdm

from ._load_db import load_db_novelty, load_db_fitness


class Condenser:
    def __init__(self, seperators:list, gen_amount:int = 400):
        self.gen_amount = gen_amount
        self.seperators = seperators

        self.novelty_averages = dict.fromkeys(self.seperators)
        self.fitness_averages = dict.fromkeys(self.seperators)


        self.fitness_data = dict.fromkeys(self.seperators)
        self.novelty_data = dict.fromkeys(self.seperators)

        for key in seperators:
            self.novelty_averages[key] = []
            self.fitness_averages[key] = []

            self.novelty_data[key] = [[] for _ in range(self.gen_amount)]
            self.fitness_data[key] = [[] for _ in range(self.gen_amount)]


    def populate_from_dir(self, db_dir:str):
        dbs = glob(f"{db_dir}/*")
        for db in tqdm(dbs):
            n_df = load_db_novelty(db)
            f_df = load_db_fitness(db)

            try:
                for seperator in self.seperators:
                    if seperator in db:
                        self._populate_fitness(f_df, seperator)
                        self._populate_novelty(n_df, seperator)
            except:
                continue

    def _populate_novelty(self, df: pd.DataFrame, seperator:str):
        vals = (df[["generation_index", "value"]].groupby(by="generation_index")["value"].apply(list))
        if len(vals) < self.gen_amount:
            raise Exception("Not fully populated df")

        vm = [0]*self.gen_amount
        for i in range(self.gen_amount):
            v = vals.iloc[i]
            vm[i] = sum(v)/len(v)
            self.novelty_data[seperator][i].extend(v)
        self.novelty_averages[seperator].append(sum(vm)/len(vm))

    def _populate_fitness(self, df: pd.DataFrame, seperator: str):
        vals = (df[["generation_index", "value"]].groupby(by="generation_index")["value"].apply(list))
        if len(vals) < self.gen_amount:
            raise Exception("Not fully populated df")

        vm = [0] * self.gen_amount
        for i in range(self.gen_amount):
            v = vals.iloc[i]
            vm[i] = sum(v) / len(v)
            self.fitness_data[seperator][i].extend(v)
        self.fitness_averages[seperator].append(sum(vm) / len(vm))




