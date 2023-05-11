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

        self.novelty_averages = dict.fromkeys(self.seperators)
        self.fitness_averages = dict.fromkeys(self.seperators)

        self.fitness_data = dict.fromkeys(self.seperators)
        self.novelty_data = dict.fromkeys(self.seperators)
        self.mmeasure_data = dict.fromkeys(self.seperators)

        for key in seperators:
            self.novelty_averages[key] = []
            self.fitness_averages[key] = []

            self.mmeasure_data[key] = ([], [], [], [], [])
            self.novelty_data[key] = [[] for _ in range(self.gen_amount)]
            self.fitness_data[key] = [[] for _ in range(self.gen_amount)]


    def populate_from_dir(self, db_dir:str, mmeasures:bool = False):
        dbs = glob(f"{db_dir}/*")
        for db in tqdm(dbs):
            n_df = load_db_novelty(db)
            f_df = load_db_fitness(db)


            if mmeasures:
                for seperator in self.seperators:
                    if seperator in db:
                        self._populate_fitness(f_df, seperator)
                        self._populate_novelty(n_df, seperator)
                        self._populate_mmeasure(n_df, seperator)
            else:
                for seperator in self.seperators:
                    if seperator in db:
                        self._populate_fitness(f_df, seperator)
                        self._populate_novelty(n_df, seperator)





    def _populate_novelty(self, df: pd.DataFrame, seperator:str):
        vals = (df[[    "generation_index", "value"]].groupby(by="generation_index")["value"].apply(list))
        if len(vals) < self.gen_amount:
            raise Exception("Not fully populated df")

        vm = [0]*self.gen_amount
        for i in range(self.gen_amount):
            v = vals.iloc[i]
            vm[i] = sum(v)/len(v)
            self.novelty_data[seperator][i].extend(v)
        self.novelty_averages[seperator].append(sum(vm)/len(vm))

    def _populate_fitness(self, df: pd.DataFrame, seperator: str):
        vals = df[["generation_index", "value"]].groupby(by="generation_index")["value"].apply(list)

        if len(vals) < self.gen_amount:
            raise Exception("Not fully populated df")

        vm = [0] * self.gen_amount
        for i in range(self.gen_amount):
            v = vals.iloc[i]
            vm[i] = sum(v) / len(v)
            self.fitness_data[seperator][i].extend(v)
        self.fitness_averages[seperator].append(sum(vm) / len(vm))

    def _populate_mmeasure(self, df:pd.DataFrame, seperator: str):
        genomes = df["serialized_multineat_genome"].loc[df["generation_index"] == df["generation_index"].max()].tolist()

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




