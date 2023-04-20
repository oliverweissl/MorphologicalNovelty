from __future__ import annotations

import numpy as np
from numpy import ndarray
from multineat import Genome
from typing import List, Tuple
from math import atan2, pi, sqrt
from revolve2.genotypes.cppnwin._genotype import Genotype
from revolve2.genotypes.cppnwin.modular_robot.body_genotype_v1 import develop_v1

from ._coordinate_ops import CoordinateOperations

#os.environ["JULIA_NUM_THREADS"] = "2"
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main  # will always be marked due to IDE error
jl.eval('include("common/calc_novelty.jl")')


class PhenotypeFramework:
    """@classmethod
    def get_bricks_hinges_amount(cls, genotype: Genotype) -> (int, int):
        #assert isinstance(genotype, (Genotype, str)), f"Error: Genotype is of type {type(genotype)}, Genotype or str expected!"
        # genotype = genotype if not isinstance(genotype, str) else cls.deserialize(genotype)
        body = develop_v1(genotype)
        bricks, hinges = cls._body_to_sorted_coordinates(body)
        return len(bricks), len(hinges)"""

    @classmethod
    def get_novelty_population(cls, genotypes: List[Genotype]) -> List[float]:
        """
        calculates novelty across population.
        :param genotypes: List[Genotype | str] --> list of genotypes for population.
        :param normalization: None -> no normalization | "clipping" -> between 0,1 | "log" -> log of x
        :param test: which test to compare histograms (chi-squared, yates-chi-squared)
        :return: List[float] novelty rate per individual
        """
        amt_instances = len(genotypes)
        bodies = [develop_v1(genotype) for genotype in genotypes]  # db only returns Genotypes, can be swithced to str using cls.deserialize()

        # TODO: check whats better + test if works
        coords = CoordinateOperations.coords_from_bodies(bodies=bodies, cob_heuristics=True)

        hists = [None] * amt_instances
        i = 0
        for coord in coords:
            mag, orient = cls._coordinates_to_magnitudes_orientation(coord)
            hists[i] = cls._gen_gradient_histogram(magnitudes=mag, orientations=orient)
            i += 1

        # This takes most computation -> in python: ~ 63 sec, julia: ~ 1.1 sec
        novelty_scores = Main.calculate_novelty(hists)
        mx_score = max(novelty_scores)
        novelty_scores = [float(score/mx_score) for score in novelty_scores] # if score > 0. else 0.
        # scaling because the min novelty is 0 in theory --> some populations can have no duplicates therefore no 0s
        return novelty_scores

    @classmethod
    def deserialize(cls, serialized_genotype: str):
        genotype = Genotype(Genome())
        genotype.genotype.Deserialize(serialized_genotype)
        return genotype

    @classmethod
    def _coordinates_to_magnitudes_orientation(cls, coordinates: ndarray) -> Tuple[
        List[float], List[Tuple[float, float]]]:
        mags = [0] * len(coordinates)  # faster than append
        orient = [(0, 0)] * len(coordinates)  # faster than append
        i = 0  # faster than enumerate
        for coord in coordinates:
            if len(coord) == 3:
                ax = atan2(sqrt(coord[1] ** 2 + coord[2] ** 2), coord[0]) * 180 / pi
                az = atan2(coord[2], sqrt(coord[1] ** 2 + coord[0] ** 2)) * 180 / pi
                orient[i] = (ax, az)
                mags[i] = np.sqrt(coord.dot(coord))
            i += 1
        return mags, orient

    @classmethod
    def _gen_gradient_histogram(cls, magnitudes: List[float], orientations: List[Tuple[float, float]], num_bins: int = 20) -> ndarray:

        bin_size = int(360 / num_bins)
        assert bin_size == 360 / num_bins, "Error: num_bins has to be a divisor of 360"

        hist = np.zeros((num_bins, num_bins), dtype=float)
        for rot, mag in zip(orientations, magnitudes):
            x, z = cls._get_bin_idx(rot, bin_size)
            hist[x][z] += mag

        hist = cls._wasserstein_softmax(hist)
        return hist

    @classmethod
    def _get_bin_idx(cls, orientation: Tuple[float, float], bin_size) -> (int, int):
        return int(orientation[0] / bin_size), int(orientation[1] / bin_size)

    @staticmethod
    def _wasserstein_softmax(arr: ndarray) -> ndarray:
        arr += (1/arr.size)
        norm = np.true_divide(arr, arr.sum())
        return norm
