from glob import glob
from numpy import std
import pandas as pd
import zipfile
from tqdm import tqdm
from common import PhenotypeFramework as pf
from revolve2.genotypes.cppnwin.modular_robot.body_genotype_v1 import develop_v1
from revolve2.core.modular_robot import MorphologicalMeasures

from ._load_db import load_db_novelty, load_db_fitness


class _Metric:
    def __init__(self, seperators:list, gen_amount:int = 400):
        self.gen_amount = gen_amount
        self.seperators = seperators

        self.data = dict.fromkeys(self.seperators)
        self.mean = dict.fromkeys(self.seperators)
        self.std = dict.fromkeys(self.seperators)
        self.min = dict.fromkeys(self.seperators)
        self.max = dict.fromkeys(self.seperators)

        for key in seperators:
            self.data[key] = [[] for _ in range(self.gen_amount)]
            self.mean[key] = [[] for _ in range(self.gen_amount)]
            self.std[key] = [[] for _ in range(self.gen_amount)]
            self.min[key] = [[] for _ in range(self.gen_amount)]
            self.max[key] = [[] for _ in range(self.gen_amount)]


class Condenser:
    def __init__(self, seperators:list, gen_amount:int = 400, extract_path:str=None):
        if extract_path:
            self._extract(seperators, extract_path)
        self.gen_amount = gen_amount
        self.seperators = seperators

        self.fitness = _Metric(self.seperators, self.gen_amount)
        self.novelty = _Metric(self.seperators, self.gen_amount)


        self.mmeasure_data = dict.fromkeys(self.seperators)
        self.best_final_morphologies = dict.fromkeys(self.seperators)
        self.best_final_fitnesses = dict.fromkeys(self.seperators)

        for key in seperators:
            self.mmeasure_data[key] = ([], [], [], [], [])
            self.best_final_morphologies[key] = []
            self.best_final_fitnesses[key] = []


    def populate_from_dir(self, db_dir:str):
        dbs = glob(f"{db_dir}/*")
        for db in tqdm(dbs):
            n_df = load_db_novelty(db)
            f_df = load_db_fitness(db)

            for seperator in self.seperators:
                if seperator in db:
                    self._populate_metric(self.fitness, f_df, seperator)
                    self._populate_metric(self.novelty, n_df, seperator)
                    self._populate_genotypes(f_df, seperator)
                    self._populate_mmeasure(n_df, seperator)

    def _populate_metric(self, metric:_Metric, df:pd.DataFrame, seperator:str):
        vals = (df[["generation_index", "value"]].groupby(by="generation_index")["value"].apply(list))
        if len(vals) < self.gen_amount:
            raise Exception("Not fully populated df")

        for i in range(self.gen_amount):
            v = vals.iloc[i]
            metric.data[seperator][i].extend(v)
            metric.mean[seperator][i].append(sum(v)/len(v))
            metric.min[seperator][i].append(min(v))
            metric.max[seperator][i].append(max(v))
            metric.std[seperator][i].append(std(v))

    def _populate_genotypes(self, df: pd.DataFrame, seperator:str, n: int = 5):
        df = df[df["generation_index"] == max(df["generation_index"])]
        vals = df[["serialized_multineat_genome", "value"]]
        genotypes = vals["serialized_multineat_genome"].values
        values = vals["value"].values

        idx = sorted(range(len(values)), key=lambda i: values[i])[-n:]
        self.best_final_morphologies[seperator].extend(genotypes[idx])
        self.best_final_fitnesses[seperator].extend(values[idx])

    def _populate_mmeasure(self, df:pd.DataFrame, seperator: str):
        genomes = df["serialized_multineat_genome"].loc[df["generation_index"] == df["generation_index"].max()].tolist()
        #genomes = self.genotype_data[seperator][-1]
        for genome in genomes:
            body = develop_v1(pf.deserialize(genome))
            try:
                mm = MorphologicalMeasures(body)
                ll, c, l, s, sz = mm.length_of_limbs, mm.coverage, mm.limbs, mm.symmetry, mm.num_modules

                self.mmeasure_data[seperator][0].append(ll)
                self.mmeasure_data[seperator][1].append(c)
                self.mmeasure_data[seperator][2].append(l)
                self.mmeasure_data[seperator][3].append(s)
                self.mmeasure_data[seperator][4].append(sz)
            except:
                pass

    def _extract(self, seperators, path):
        name = path.split(".zip")[0]
        with zipfile.ZipFile(path) as archive:
            for file in archive.namelist():
                for seperator in seperators:
                    if f"db_n_{seperator}" in file:
                        archive.extract(file, name)
