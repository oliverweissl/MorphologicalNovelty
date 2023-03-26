from __future__ import annotations

import numpy as np

from multineat import Genome
from typing import List, Tuple
from math import atan2, pi, sqrt

from numpy import ndarray

from .compare_histograms import CompareHistorgrams as ch
from revolve2.genotypes.cppnwin._genotype import Genotype
from revolve2.core.modular_robot import Body, ActiveHinge, Brick
from revolve2.genotypes.cppnwin.modular_robot.body_genotype_v1 import develop_v1


class PhenotypeFramework:

    @classmethod
    def get_bricks_hinges_amount(cls, genotype: Genotype | str) -> (int, int):
        assert isinstance(genotype, (Genotype, str)), f"Error: Genotype is of type {type(genotype)}, Genotype or str expected!"

        genotype = genotype if not isinstance(genotype, str) else cls.deserialize(genotype)
        body = develop_v1(genotype)
        bricks, hinges = cls._body_to_sorted_coordinates(body)
        return len(bricks), len(hinges)

    @classmethod
    def get_novelty_population(cls, genotypes: List[Genotype | str], novelty_test: Tuple[str|None, float|None],  normalization: str = None) -> List[float]:
        """
        calculates novelty across population.
        :param genotypes: List[Genotype | str] --> list of genotypes for population.
        :param normalization: None -> no normalization | "clipping" -> between 0,1 | "log" -> log of x
        :param test: which test to compare histograms (chi-squared, yates-chi-squared)
        :return: List[float] novelty rate per individual
        """
        amt_instances = len(genotypes)

        genotypes = [genotype if not isinstance(genotype, str) else cls.deserialize(genotype) for genotype in genotypes]
        bodies = [develop_v1(genotype) for genotype in genotypes]

        coords = [cls._body_to_sorted_coordinates(body) for body in bodies]
        coords = [cls._coordinates_pca_change_basis(coord) for coord in coords] # PCA change of basis -> orientation of variance/ covariance


        brick_hists = [None] * amt_instances
        hinge_hists = [None] * amt_instances

        i = 0
        for coord in coords:
            brick_mag, brick_orient = cls._coordinates_to_magnitudes_orientation(coord[0])
            brick_hists[i] = cls._gen_gradient_histogram(magnitudes=brick_mag,
                                                         orientations=brick_orient,
                                                         normalization=normalization)

            hinge_mag, hinge_orient = cls._coordinates_to_magnitudes_orientation(coord[1])
            hinge_hists[i] = cls._gen_gradient_histogram(magnitudes=hinge_mag,
                                                         orientations=hinge_orient,
                                                         normalization=normalization)
            i += 1

        brick_novelty_scores = [0] * amt_instances
        hinge_novelty_scores = [0] * amt_instances
        for i in range(amt_instances - 1):
            for j in range(i + 1, amt_instances):
                brick_score = cls._compare_hist(brick_hists[i], brick_hists[j], novelty_test)
                brick_novelty_scores[i] += brick_score
                brick_novelty_scores[j] += brick_score

                hinge_score = cls._compare_hist(hinge_hists[i], hinge_hists[j], novelty_test)
                hinge_novelty_scores[i] += hinge_score
                hinge_novelty_scores[j] += hinge_score
                #print(f"Instances: {i}, {j}: brick_score: {brick_score}, hinge_score:  {hinge_score}")

        novelty_scores = [(b_score + h_score) / 2
                          for b_score, h_score in zip(brick_novelty_scores, hinge_novelty_scores)]


        novelty_scores = cls._normalize_list(novelty_scores, "scaling")
        # scaling because the min novelty is 0 in theory --> some populations can have no duplicates therefore no 0s
        return novelty_scores.tolist()

    @classmethod
    def deserialize(cls, serialized_genotype: str):
        genotype = Genotype(Genome())
        genotype.genotype.Deserialize(serialized_genotype)
        return genotype

    @classmethod
    def _compare_hist(cls, O: List[List[float]], E: List[List[float]], novelty_test: Tuple[str, float|None]) -> float:
        test, p = novelty_test
        assert (test is not None) or (p is not None), "ERROR: no test or p value given"
        # selects test functions
        if p is not None:
            score = ch.minkowsky_distance(O, E, p)
        else:
            score = {'yates-chi-squared': ch.yates_chi_squared,
                     'chi-squared': ch.chi_squared,
                     'hellinger-dist': ch.hellinger_distance,
                     'manhattan-dist': ch.manhattan_distance,
                     'euclidian-dist': ch.euclidian_distance,
                     'chybyshev-dist': ch.chybyshev_distance,
                     'pcc': ch.pearsons_correlation_coefficient,
                     }[test](O, E)
        return score

    @classmethod
    def _body_to_sorted_coordinates(cls, body: Body) -> (List[Tuple], List[Tuple]):
        """
        Generates coordinates from Body object. All resulting coordinates are normalized to (0,0,0)
        --> core is forced to the origin.

        :param body:
        :return: coordinates of bricks, coordinates of hinges --> (x,y,z)
        """
        body_arr, core_pos = body.to_grid()
        body_arr = np.asarray(body_arr)

        x_size, y_size, z_size = body_arr.shape


        flat_body_arr = body_arr.flatten()
        flat_body_coords = np.asarray([[[
            tuple(np.subtract((x, y, z), core_pos)) for x in range(x_size)]
            for y in range(y_size)]
            for z in range(z_size)], dtype="i,i,i").flatten()

        bricks, hinges = [], []
        for elem, coord in zip(flat_body_arr, flat_body_coords):
            if isinstance(elem, ActiveHinge):
                hinges.append(coord)
            elif isinstance(elem, Brick):
                bricks.append(coord)

        return bricks, hinges

    @classmethod
    def _coordinates_pca_change_basis(cls, coords: Tuple[List[Tuple], List[Tuple]]) -> (ndarray, ndarray):
        bricks, hinges = coords
        bricks = np.asarray([list(b) for b in bricks])
        hinges = np.asarray([list(h) for h in hinges])

        all_coords = []
        [all_coords.append(val) for val in bricks if len(bricks) > 0]
        [all_coords.append(val) for val in hinges if len(hinges) > 0]

        if len(all_coords) > 1: #has to be more than 1 block, otherwise np.cov doesnt work
            all_coords = np.asarray(all_coords).T

            A = np.cov(all_coords)
            e, v = np.linalg.eig(np.dot(A, A.T)/(len(all_coords)-1)) # eigenvalues, eigenvectors
            srt = np.argsort(-e) #sorting axis, x-axis: biggest variance, y-axis second biggest, z-axis:smallest
            inv_sorted_v = np.linalg.inv(v[srt].T)

            bricks = np.dot(inv_sorted_v, bricks.T) if len(bricks) > 0 else bricks
            hinges = np.dot(inv_sorted_v, hinges.T) if len(hinges) > 0 else hinges
            bricks, hinges = bricks.T, hinges.T



        return bricks, hinges

    @classmethod
    def _coordinates_to_magnitudes_orientation(cls, coordinates: List[Tuple]) -> Tuple[List[float], List[Tuple[float, float]]]:
        mags = [None] * len(coordinates)  # faster than append
        orient = [None] * len(coordinates)  # faster than append
        i = 0  # faster than enumerate
        for elem in coordinates:
            elem = list(elem)
            ax = atan2(sqrt(elem[1] ** 2 + elem[2] ** 2), elem[0]) * 180 / pi
            az = atan2(elem[2], sqrt(elem[1] ** 2 + elem[0] ** 2)) * 180 / pi
            orient[i] = (ax, az)

            elem = np.asarray(elem)
            mags[i] = np.sqrt(elem.dot(elem))
            i += 1
        return mags, orient

    @classmethod
    def _gen_gradient_histogram(cls, magnitudes: List[float], orientations: List[Tuple[float, float]], normalization: str | None, num_bins: int = 18) -> ndarray:
        """
        Generates a 2D-Historgram of oriented gradients, using bins to standardize feature size. Can be normalized in various ways to make it comparable.
        :param magnitudes: Magnitudes List[float]
        :param orientations: Rotations List[Tuple[float,float]] -> tuple (rotation_x, rotation:z)
        :param normalization: None -> no normalization | "clipping" -> between 0,1 | "log" -> log of x
        :param num_bins: Number of bins to divide --> the higher the more detail
        :return:
        """

        def _get_bin_idx(orientation: Tuple[float, float], bin_size) -> (int, int):
            return int(orientation[0] / bin_size), int(orientation[1] / bin_size)

        bin_size = int(360 / num_bins)
        assert bin_size == 360 / num_bins, "Error: num_bins has to be a divisor of 360"

        hist = [[0] * num_bins for _ in range(num_bins)]  # faster than with numpy
        for rot, mag in zip(orientations, magnitudes):
            x, z = _get_bin_idx(rot, bin_size)
            hist[x][z] += mag

        hist = cls._normalize_list(hist, normalization)
        return hist

    @staticmethod
    def _normalize_list(lst: list | np.array, normalization: str | None) -> ndarray:
        new_lst = np.asarray(lst)
        max_val, min_val = new_lst.max(), new_lst.min()

        if not normalization:
            norm = lambda x: x
        elif "clipping" in normalization:
            norm = lambda x: (x - min_val) / (max_val - min_val) if max_val != min_val else x
        elif "log" in normalization:
            norm = lambda x: np.ma.log(x).filled(0)
        elif "scaling" in normalization:
            norm = lambda x: x / max_val if max_val != min_val else x

        normalized_array = norm(lst)
        return normalized_array
