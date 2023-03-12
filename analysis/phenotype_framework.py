from __future__ import annotations

import numpy as np
from multineat import Genome
from typing import List, Tuple
from math import atan2, pi, sqrt
from revolve2.genotypes.cppnwin._genotype import Genotype
from revolve2.core.modular_robot import Body, ActiveHinge, Brick
from revolve2.genotypes.cppnwin.modular_robot.body_genotype_v1 import develop_v1


class PhenotypeFramework:

    @classmethod
    def get_blocks_hinges_amount(cls, genotype: Genotype | str) -> (int, int):
        assert isinstance(genotype, (Genotype, str)), f"Error: Genotype is of type {type(genotype)}, Genotype or str expected!"
        genotype = genotype if not isinstance(genotype, str) else cls.deserialize(genotype)
        body = develop_v1(genotype)
        bricks, hinges = cls._body_to_sorted_coordinates(body)
        return len(bricks), len(hinges)

    @classmethod
    def get_novelty_population(cls, genotypes: List[Genotype | str], normalization:str) -> List[float]:
        """
        calculates novelty across population.
        :param genotypes: List[Genotype | str] --> list of genotypes for population.
        :param normalization: None -> no normalization | "clipping" -> between 0,1 | "log" -> log of x
        :return: List[float] novelty rate per individual
        """
        amt_instances = len(genotypes)

        genotypes = [genotype if not isinstance(genotype, str) else cls.deserialize(genotype) for genotype in genotypes]
        bodies = [develop_v1(genotype) for genotype in genotypes]
        coords = [cls._body_to_sorted_coordinates(body) for body in bodies]

        hists = [None]*amt_instances
        i = 0
        for coord in coords:
            hists[i] = cls._gen_gradient_histogram(magnitudes= cls._coordinates_to_magnitudes(coord),
                                                   orientations=cls._coordinates_to_orientation(coord),
                                                   normalization= normalization)
            i += 1







    @classmethod
    def deserialize(cls, serialized_genotype: str):
        genotype = Genotype(Genome())
        genotype.genotype.Deserialize(serialized_genotype)
        return genotype

    @classmethod
    def _body_to_sorted_coordinates(cls, body: Body) -> (List[Tuple],List[Tuple]):
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
            tuple(np.subtract((x,y,z),core_pos)) for x in range(x_size)]
            for y in range(y_size)]
            for z in range(z_size)],dtype="i,i,i").flatten()

        bricks, hinges = [], []
        for elem, coord in zip(flat_body_arr, flat_body_coords):
            if isinstance(elem, ActiveHinge):
                hinges.append(coord)
            elif isinstance(elem, Brick):
                bricks.append(coord)

        return bricks, hinges

    @classmethod
    def _coordinates_to_magnitudes(cls, coordinates: List[Tuple]) -> List[float]:
        """
        Returns array of vector magnitudes for array of coordinate tuples.
        Accepts List[Tuple] Tuple -> (x,y,z)
        :param coordinates:
        :return: List of magnitudes (Float)
        """
        mags = [None] * len(coordinates)  # faster than append
        i = 0  # faster than enumerate
        for elem in coordinates:
            elem = np.asarray(list(elem))
            mags[i] = np.sqrt(elem.dot(elem))
            i += 1
        return mags

    @classmethod
    def _coordinates_to_orientation(cls, coordinates: List[Tuple]) -> List[Tuple[float, float]]:
        """
        Calcuclates the orientation of a vector with respect to the x and z axis.
        y-axis is obsolete, since its x- 90deg.
        Results in a tuple of (angle-x, angle-y), measured in radiants.
        :param coordinates:
        :return:
        """
        i = 0  # faster than enumerate
        rot_vec = [None] * len(coordinates)  # faster than append
        for elem in coordinates:
            elem = list(elem)
            # math faster than np in this case
            ax = atan2(sqrt(elem[1] ** 2 + elem[2] ** 2), elem[0]) * 180 / pi
            az = atan2(elem[2], sqrt(elem[1] ** 2 + elem[0] ** 2)) * 180 / pi

            rot_vec[i] = (ax, az)
            i += 1
        return rot_vec

    @classmethod
    def _gen_gradient_histogram(cls, magnitudes: List[float],
                                orientations: List[Tuple[float, float]],
                                normalization: str,
                                num_bins: int = 18) -> List[List[float]]:
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

        if not normalization:
            return hist

        nphist = np.asarray(hist)
        if normalization == "clipping":
            max_val, min_val = nphist.max(), nphist.min()
            norm = lambda x: (x - min_val) / (max_val - min_val)
            hist = norm(nphist)

        if normalization == "log":
            hist = np.where(nphist != 0, np.log(nphist), 0)

        return hist