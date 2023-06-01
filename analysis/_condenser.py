from glob import glob
import pandas as pd
from tqdm import tqdm
from common import PhenotypeFramework as pf
from revolve2.genotypes.cppnwin.modular_robot.body_genotype_v1 import develop_v1
from revolve2.core.modular_robot import MorphologicalMeasures

from ._load_db import load_db_novelty, load_db_fitness


class Condenser:
    def __init__(self, seperators:list, gen_amount:int = 400):
        self.gen_amount = gen_amount
        self.seperators = seperators

        self.fitness_data = dict.fromkeys(self.seperators)
        self.fitness_std = dict.fromkeys(self.seperators)
        self.fitness_max = dict.fromkeys(self.seperators)

        self.novelty_data = dict.fromkeys(self.seperators)
        self.novelty_std = dict.fromkeys(self.seperators)
        self.novelty_min = dict.fromkeys(self.seperators)

        self.mmeasure_data = dict.fromkeys(self.seperators)

        self.best_final_morpholigeis = dict.fromkeys(self.seperators)
        self.glob_max_indx = dict.fromkeys(self.seperators)

        for key in seperators:
            self.mmeasure_data[key] = ([], [], [], [], [])

            self.novelty_data[key] = [[] for _ in range(self.gen_amount)]
            self.novelty_min[key] = [[] for _ in range(self.gen_amount)]
            self.novelty_std[key] = [[] for _ in range(self.gen_amount)]


            self.fitness_data[key] = [[] for _ in range(self.gen_amount)]
            self.fitness_max[key] = [[] for _ in range(self.gen_amount)]
            self.fitness_std[key] = [[] for _ in range(self.gen_amount)]

            self.best_final_morpholigeis[key] = []


    def populate_from_dir(self, db_dir:str):
        dbs = glob(f"{db_dir}/*")
        for db in tqdm(dbs):
            n_df = load_db_novelty(db)
            f_df = load_db_fitness(db)

            for seperator in self.seperators:
                if seperator in db:
                    self._populate_fitness(f_df, seperator)
                    self._populate_novelty(n_df, seperator)
                    #self._populate_genotypes(f_df, seperator)
                    self._populate_mmeasure(n_df, seperator)

    def _populate_novelty(self, df: pd.DataFrame, seperator:str):
        vals = (df[["generation_index", "value"]].groupby(by="generation_index")["value"].apply(list))
        if len(vals) < self.gen_amount:
            raise Exception("Not fully populated df")

        for i in range(self.gen_amount):
            v = vals.iloc[i]
            mean = sum(v)/len(v)
            self.novelty_data[seperator][i].append(mean)
            self.novelty_min[seperator][i].append(min(v))

    def _populate_fitness(self, df: pd.DataFrame, seperator: str):
        vals = df[["generation_index", "value"]].groupby(by="generation_index")["value"].apply(list)

        if len(vals) < self.gen_amount:
            raise Exception("Not fully populated df")

        for i in range(self.gen_amount):
            v = vals.iloc[i]
            mean = sum(v) / len(v)
            self.fitness_data[seperator][i].append(mean)
            self.fitness_max[seperator][i].append(max(v))
            self.fitness_std[seperator][i].append()

    def _populate_genotypes(self, df: pd.DataFrame, seperator:str):
        vals = (df[["generation_index", "serialized_multineat_genome"]].groupby(by="generation_index")[
                    "serialized_multineat_genome"].apply(list))
        if len(vals) < self.gen_amount:
            raise Exception("Not fully populated df")
        for i in range(self.gen_amount):
            self.genotype_data[seperator][i].extend(vals.iloc[i])

    def _populate_mmeasure(self, df:pd.DataFrame, seperator: str):
        genomes = df["serialized_multineat_genome"].loc[df["generation_index"] == df["generation_index"].max()].tolist()
        #genomes = self.genotype_data[seperator][-1]
        for genome in genomes:
            body = develop_v1(pf.deserialize(genome))
            try:
                mm = MorphologicalMeasures(body)
                ll,c,l,s,sz = mm.length_of_limbs, mm.coverage, mm.limbs, mm.symmetry, mm.num_modules

                self.mmeasure_data[seperator][0].append(ll)
                self.mmeasure_data[seperator][1].append(c)
                self.mmeasure_data[seperator][2].append(l)
                self.mmeasure_data[seperator][3].append(s)
                self.mmeasure_data[seperator][4].append(sz)
            except:
                pass




