from abc import abstractmethod
from typing import List
import numpy as np
from itertools import chain
import time
from math import sqrt
from numpy import ndarray

INT_CASTER = 10_000


class CompareHistorgrams:

    @classmethod
    def yates_chi_squared(cls, O: List[List[float]], E: List[List[float]]) -> float:
        itr = len(O)
        assert itr == len(E), f"Error: Histograms have different sizes -> O:{itr}, E:{len(E)}"

        score = 0
        for o, e in zip(O,E):
            for vo, ve in zip(o,e):
                ve += 1e-6
                score += ((abs(vo-ve)-0.5)**2)/ve
        return score

    @classmethod
    def chi_squared(cls, O: List[List[float]], E: List[List[float]]) -> float:
        itr = len(O)
        assert itr == len(E), f"Error: Histograms have different sizes -> O:{itr}, E:{len(E)}"

        score = 0
        for o, e in zip(O, E):
            for vo, ve in zip(o, e):
                ve += 1e-6
                score += (abs(vo - ve) ** 2) / ve
        return score

    @classmethod
    def hellinger_distance(cls, O: List[List[float]], E: List[List[float]]) -> float:
        itr = len(O)
        assert itr == len(E), f"Error: Histograms have different sizes -> O:{itr}, E:{len(E)}"

        score = 0
        for o, e in zip(O, E):
            for vo, ve in zip(o, e):
                score += (sqrt(vo)-sqrt(ve))**2
        score = 1/sqrt(2)*sqrt(score)
        return score

    @classmethod
    def manhattan_distance(cls, O: List[List[float]], E: List[List[float]]) -> float:
        itr = len(O)
        assert itr == len(E), f"Error: Histograms have different sizes -> O:{itr}, E:{len(E)}"

        score = 0
        for o, e in zip(O, E):
            for vo, ve in zip(o, e):
                score += abs(vo-ve)
        return score

    @classmethod
    def euclidian_distance(cls, O: List[List[float]], E: List[List[float]]) -> float:
        itr = len(O)
        assert itr == len(E), f"Error: Histograms have different sizes -> O:{itr}, E:{len(E)}"

        score = 0
        for o, e in zip(O, E):
            for vo, ve in zip(o, e):
                score += (vo-ve)**2
        score = sqrt(score)
        return score

    @classmethod
    def chybyshev_distance(cls, O: List[List[float]], E: List[List[float]]) -> float:
        itr = len(O)
        assert itr == len(E), f"Error: Histograms have different sizes -> O:{itr}, E:{len(E)}"

        score = 0
        for o, e in zip(O, E):
            for vo, ve in zip(o, e):
                canditade_score = abs(vo-ve)
                score = canditade_score if canditade_score > score else score

        return score

    @classmethod
    def minkowsky_distance(cls, O: List[List[float]], E: List[List[float]], p: int) -> float:
        assert 0 < p, f"Error: p has to be bigger than 0 for this implementation"
        itr = len(O)
        assert itr == len(E), f"Error: Histograms have different sizes -> O:{itr}, E:{len(E)}"

        score = 0
        for o, e in zip(O, E):
            for vo, ve in zip(o, e):
                score += abs(vo-ve)**p
        score = score**(1/p)
        return score

    @classmethod
    def pearsons_correlation_coefficient(cls, O: List[List[float]], E: List[List[float]]) -> float:
        itr = len(O)
        assert itr == len(E), f"Error: Histograms have different sizes -> O:{itr}, E:{len(E)}"

        upper = 0
        lower_o, lower_e = 0, 0

        flat_O = list(chain.from_iterable(O))
        flat_E = list(chain.from_iterable(E))

        avg_o = sum(flat_O)/len(flat_O)
        avg_e = sum(flat_E)/len(flat_E)

        for o, e in zip(O, E):
            for vo, ve in zip(o, e):
                vo += 1e-6
                ve += 1e-6

                upper += (vo-avg_o)*(ve-avg_e)
                lower_o += (vo-avg_o)**2
                lower_e += (ve-avg_e)**2

        lower_o = sqrt(lower_o)
        lower_e = sqrt(lower_e)

        score = upper/(lower_o*lower_e)
        return score

    @classmethod
    def wasserstein_dist(cls, O: List[List[float]], E: List[List[float]]) -> float:
        """
        the wasserstein distance quantifies, how similar two distributions are.
        This implementation uses a INT_CASTER, to mitigate floating point calculation errors.
        A distribution needs to sum up to 1 for the algorithm to finnish.
        --> the INT_CASTER is a multiplicator of such
        --> casting to int leaves margin -> margin is added back as random noise on the dist.

        literature: https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/rubner-jcviu-00.pdf
        :param O: a distribution of size NxN
        :param E: a distribution of size NxN
        :return: float value of novelty score
        """

        O, E = np.asarray(O), np.asarray(E)
        assert O.shape == E.shape, f"Error: Histograms have different sizes -> O:{O.shape}, E:{E.shape}"
        assert round(O.sum()) == round(E.sum()) == 1., f"Error: Distributions  dont sum up to 1 -> O:{round(O.sum())}, E:{round(E.sum())}"
        xsize, ysize = O.shape

        supply, capacity = np.copy(O*INT_CASTER).astype(int), np.copy(E*INT_CASTER).astype(int)
        supply, capacity = cls._apply_noise_mask(supply), cls._apply_noise_mask(capacity)

        score = 0
        while True:
            from_idx = (supply != 0).argmax()
            fx, fy = int(from_idx / ysize), from_idx % xsize

            to_idx = (capacity != 0).argmax()
            tx, ty = int(to_idx / ysize), to_idx % xsize

            if supply[fx][fy] == 0 or capacity[tx][ty] == 0:
                break

            work, supply, capacity = cls._move_supply(supply, fx, fy, capacity, tx, ty)
            score += work
        return float(score)

    @classmethod
    def _apply_noise_mask(cls, dist: ndarray) -> ndarray:
        xsize, ysize = dist.shape
        diff = INT_CASTER - dist.sum()
        mask = np.zeros(xsize * ysize)
        mask[:diff] = 1
        np.random.shuffle(mask)

        mask = mask.reshape((xsize, ysize))

        dist = np.add(dist, mask)
        assert dist.sum() == INT_CASTER, f"ERROR: Sum of Dist is expected to be: {INT_CASTER} ,but is: {dist.sum()}"
        return dist

    @staticmethod
    def _move_supply(selected_supply: ndarray, sx: int, sy: int,
                     selected_capacity: ndarray, cx: int, cy: int) -> (float, ndarray, ndarray):

        if selected_supply[sx][sy] <= selected_capacity[cx][cy]:
            flow = selected_supply[sx][sy]
            selected_capacity[cx][cy] -= flow
            selected_supply[sx][sy] = 0
        else:
            flow = selected_capacity[cx][cy]
            selected_supply[sx][sy] -= flow
            selected_capacity[cx][cy] = 0
        distance = sqrt(np.abs(sx - cx) ** 2 + np.abs(sy - cy) ** 2)
        return flow/INT_CASTER*distance, selected_supply, selected_capacity

    @abstractmethod
    def example(cls, O: List[List[float]], E: List[List[float]]) -> float:
        itr = len(O)
        assert itr == len(E), f"Error: Histograms have different sizes -> O:{itr}, E:{len(E)}"

        score = 0
        for o, e in zip(O, E):
            for vo, ve in zip(o, e):
                print("hello")
        #TODO: implement
        return score