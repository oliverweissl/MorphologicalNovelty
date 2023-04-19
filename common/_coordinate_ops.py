import numpy as np
from numpy import ndarray
from typing import List, Tuple
from math import atan2, pi, sqrt
from scipy.spatial.transform import Rotation as R
from revolve2.core.modular_robot import Body, Brick, ActiveHinge


class CoordinateOperations:

    @classmethod
    def coords_from_bodies(cls, bodies: List[Body], cob_heuristics: bool = True) -> List:
        coords = [cls._body_to_sorted_coordinates(body) for body in bodies]
        if cob_heuristics:
            return [cls._coordinates_pca_heuristic(coord) for coord in coords]
        else:
            return [cls._coordinates_pca_change_basis(coord) for coord in coords]

    @classmethod
    def _body_to_sorted_coordinates(cls, body: Body) -> ndarray:
        body_arr, core_pos = body.to_grid()
        body_arr = np.asarray(body_arr)

        x, y, z = body_arr.shape

        elems = []
        for xe in range(x):
            for ye in range(y):
                for ze in range(z):
                    elem = body_arr[xe][ye][ze]
                    if isinstance(elem, ActiveHinge) or isinstance(elem, Brick):
                        elems.append(np.subtract((xe, ye, ze), core_pos))
        elems = np.asarray(elems)
        return elems

    @classmethod
    def _coordinates_pca_change_basis(cls, coords: ndarray) -> ndarray:
        if len(coords) > 1:
            covariance_matrix = np.cov(coords.T)
            eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

            srt = np.argsort(eigen_values)[
                  ::-1]  # sorting axis, x-axis: biggest variance, y-axis second biggest, z-axis:smallest
            rot_rad = np.radians(180)
            for i in range(len(srt)):
                while True:
                    if srt[i] == i:
                        break
                    cand = srt[i]
                    rotation = R.from_rotvec(rot_rad * eigen_vectors[cand])
                    coords = rotation.apply(coords)

                    eigen_vectors[i], eigen_vectors[cand] = np.copy(eigen_vectors[cand]), np.copy(eigen_vectors[i])
                    srt[[i, cand]] = srt[[cand, i]]

            coords = np.linalg.inv(eigen_vectors).dot(coords.T)
        return coords.T

    @classmethod
    def _coordinates_pca_heuristic(cls, coords: ndarray) -> ndarray:
        if len(coords) > 1:
            covariance_matrix = np.cov(coords.T)
            eigen_values, _ = np.linalg.eig(covariance_matrix)
            srt = np.argsort(eigen_values)[::-1]
            for i in range(len(srt)):
                while True:
                    if srt[i] == i:
                        break
                    cand = srt[i]
                    coords[:, [i, cand]] = coords[:, [cand, i]]
                    srt[[i, cand]] = srt[[cand, i]]
        return coords

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
