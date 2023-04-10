from __future__ import annotations

import numpy as np

from numpy import ndarray
from multineat import Genome
from typing import List, Tuple
from math import atan2, pi, sqrt

from ._compare_histograms import CompareHistorgrams as ch
from revolve2.genotypes.cppnwin._genotype import Genotype
from revolve2.core.modular_robot import Body, Brick, ActiveHinge
from revolve2.genotypes.cppnwin.modular_robot.body_genotype_v1 import develop_v1


import os
os.environ["JULIA_NUM_THREADS"] = "2"
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main  # will always be marked due to IDE error
jl.eval('include("common/calc_novelty.jl")')


class PhenotypeFramework:
    @classmethod
    def get_bricks_hinges_amount(cls, genotype: Genotype | str) -> (int, int):
        assert isinstance(genotype, (Genotype, str)), f"Error: Genotype is of type {type(genotype)}, Genotype or str expected!"

        genotype = genotype if not isinstance(genotype, str) else cls.deserialize(genotype)
        body = develop_v1(genotype)
        bricks, hinges = cls._body_to_sorted_coordinates(body)
        return len(bricks), len(hinges)

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


        coords = [cls._coordinates_pca_change_basis(cls._body_to_sorted_coordinates(body)) for body in bodies]  # PCA change of basis -> orientation of variance/ covariance

        brick_hists, hinge_hists = [None] * amt_instances, [None] * amt_instances
        i = 0
        for coord in coords:
            brick_mag, brick_orient = cls._coordinates_to_magnitudes_orientation(coord[0])
            brick_hists[i] = cls._gen_gradient_histogram(magnitudes=brick_mag,
                                                         orientations=brick_orient)

            hinge_mag, hinge_orient = cls._coordinates_to_magnitudes_orientation(coord[1])
            hinge_hists[i] = cls._gen_gradient_histogram(magnitudes=hinge_mag,
                                                         orientations=hinge_orient)
            i += 1

        # This takes most computation -> in python: ~ 63 sec, julia: ~ 34 sec
        brick_novelty_scores, hinge_novelty_scores = Main.get_novelties(brick_hists, hinge_hists)
        novelty_scores = [float((b_score + h_score) / 2)
                          for b_score, h_score in zip(brick_novelty_scores, hinge_novelty_scores)]
        mscore = max(novelty_scores)
        novelty_scores = [score/mscore for score in novelty_scores] # if score > 0. else 0.
        # scaling because the min novelty is 0 in theory --> some populations can have no duplicates therefore no 0s
        return novelty_scores

    @classmethod
    def deserialize(cls, serialized_genotype: str):
        genotype = Genotype(Genome())
        genotype.genotype.Deserialize(serialized_genotype)
        return genotype

    @classmethod
    def _compare_hist(cls, O: ndarray, E: ndarray) -> float:
        score = ch.wasserstein_dist(O, E)
        return score

    @classmethod
    def _body_to_sorted_coordinates(cls, body: Body) -> (ndarray, ndarray):
        body_arr, core_pos = body.to_grid()
        body_arr = np.asarray(body_arr)

        x, y, z = body_arr.shape

        bricks, hinges = [], []
        for xe in range(x):
            for ye in range(y):
                for ze in range(z):
                    elem = body_arr[xe][ye][ze]
                    if isinstance(elem, ActiveHinge):
                        hinges.append(np.subtract((xe, ye, ze), core_pos))
                    elif isinstance(elem, Brick):
                        bricks.append(np.subtract((xe, ye, ze), core_pos))

        bricks, hinges = np.asarray(bricks), np.asarray(hinges)
        return bricks, hinges

    @classmethod
    def _coordinates_pca_change_basis(cls, coords: Tuple[ndarray, ndarray]) -> (ndarray, ndarray):
        bricks, hinges = coords

        all_coords = np.copy(hinges)
        if bricks.size and hinges.size:
            all_coords = np.concatenate(coords)
        elif bricks.size:
            all_coords = np.copy(bricks)

        if len(all_coords) > 1: # covariance only works with n > 1 points
            covariance_matrix = np.cov(all_coords.T)
            eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)  # eigenvalues, eigenvectors

            srt = np.argsort(-eigen_values)  # sorting axis, x-axis: biggest variance, y-axis second biggest, z-axis:smallest
            inv_sorted_vectors = np.linalg.inv(eigen_vectors[srt].T)

            bricks = np.dot(inv_sorted_vectors, bricks.T).T if len(bricks) > 0 else bricks
            hinges = np.dot(inv_sorted_vectors, hinges.T).T if len(hinges) > 0 else hinges
        return bricks, hinges

    @classmethod
    def _coordinates_to_magnitudes_orientation(cls, coordinates: ndarray) -> Tuple[List[float], List[Tuple[float, float]]]:
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
