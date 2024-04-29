"""
Gaussian Random Field module handles the data assimilation and EIBV calculation associated with locations.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2024-04-29

Objectives:
    1. Construct the Gaussian Random Field (GRF) kernel.
    2. Update the prior mean and covariance matrix.
    3. Assimilate in-situ data.

Methodology:
    1. Construct the GRF kernel.
        1.1. Construct the distance matrix using
            .. math::
                d_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2 + \ksi^2 (z_i - z_j)^2}
        1.2. Construct the covariance matrix.
            .. math::
                \Sigma_{ij} = \sigma^2 (1 + \eta d_{ij}) \exp(-\eta d_{ij})
    2. Update the prior mean and covariance matrix.
        2.1. Update the prior mean.
        2.2. Update the prior covariance matrix.
    3. Calculate the EIBV for given locations.
        3.1. Compute the EIBV for given locations.
"""
import numpy as np
from pykdtree.kdtree import KDTree
from shapely.geometry import Polygon, Point
from typing import Union
from scipy.stats import norm, multivariate_normal
from scipy.spatial.distance import cdist
from scipy.interpolate import RegularGridInterpolator
import joblib
import time
import os


class GRF:
    """
    Gaussian Random Field module handles the data assimilation and EIBV calculation.
    """
    def __init__(self, polygon_border: np.ndarray, polygon_obstacle: np.ndarray,  grid_size: float = .1,
                 lateral_range: float = .6, sigma: float = .2, nugget: float = .01, threshold: float = .5, ) -> None:
        """
        Set up the Gaussian Random Field (GRF) kernel.
        """
        # self.config = Config()
        self.polygon_border = polygon_border
        self.polygon_border_shapely = Polygon(polygon_border)
        self.polygon_obstacle = polygon_obstacle
        self.polygon_obstacle_shapely = Polygon(polygon_obstacle)

        """ Empirical parameters """
        self.__sigma = sigma
        self.__nugget = nugget
        self.__threshold = threshold
        self.__grid_size = grid_size
        self.__lateral_range = lateral_range
        self.__eta = 4.5 / self.__lateral_range  # decay factor
        self.__tau = np.sqrt(self.__nugget)  # measurement noise

        self.__discretize_grf_grid()
        self.__construct_covariance_matrix()
        self.__construct_prior_mean()
        self.__load_cdf_interpolator()
        self.__mu_prior = self.__mu
        self.__Sigma_prior = self.__Sigma
        self.__cnt = 0

    def __discretize_grf_grid(self) -> None:
        """
        Discretize and construct the GRF grid.
        """
        gridsize = self.__grid_size
        xv = np.arange(0, 1, gridsize)
        yv = np.arange(0, 1, gridsize)
        self.grid = []
        for i in range(len(yv)): 
            for j in range(len(xv)):
                if i % 2 == 0:
                    point = [xv[j], yv[i]]
                else: 
                    point = [xv[j] + gridsize / 2, yv[i]]
                if self.polygon_border_shapely.contains(Point(point[0], point[1])):
                    self.grid.append(point)
        self.__n_grf_grid = len(self.grid)
        self.grid = np.array(self.grid)
        self.grid_kdtree = KDTree(self.grid)

    def __construct_covariance_matrix(self) -> None:
        """
        Construct distance matrix and thus Covariance matrix for the kernel.

        Methodology:
            1. Construct the distance matrix using
                .. math::
                    d_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2 + \ksi^2 (z_i - z_j)^2}
            2. Construct the covariance matrix.
                .. math::
                    \Sigma_{ij} = \sigma^2 (1 + \eta d_{ij}) \exp(-\eta d_{ij})
        """
        dm = cdist(self.grid, self.grid)
        self.__Sigma = self.__sigma ** 2 * ((1 + self.__eta * dm) * np.exp(-self.__eta * dm))

    def __construct_prior_mean(self) -> None:
        """
        Construct prior mean for the kernel.

        Methodology:
            1. Construct the prior mean using the SINMOD dataset.
            2. Interpolate the prior mean onto the grid.

        Returns:
            None
        """
        self.__mu = self.grid[:, 0].reshape(-1, 1)

    def __construct_ground_truth(self) -> None:
        """
        Construct the ground truth field based on the prior mean and covariance matrix.
        """
        self.__mu_truth = self.__mu + np.linalg.cholesky(self.__Sigma) @ np.random.randn(self.__n_grf_grid, 1)

    def __load_cdf_interpolator(self) -> None:
        t1 = time.time()
        self.__rho_values, self.__z1_values, self.__z2_values, self.__cdf_values = joblib.load(
            os.getcwd() + "/interpolator_medium.joblib")
        self.__interpolators = [
            RegularGridInterpolator((self.__z1_values, self.__z2_values), self.__cdf_values[i, :, :],
                                    bounds_error=False, fill_value=None) for i in range(self.__rho_values.size)]
        t2 = time.time()
        print("Loading interpolators finished, time cost: {:.2f} s".format(t2 - t1))

    def assimilate_data(self, dataset: np.ndarray) -> tuple:
        """
        Assimilate dataset to GRF kernel.

        Args:
            dataset: np.array([x, y, sal])

        Methodology:
            1. Find the index using KDTree.
            2. Average the values to each cell.
            3. Update the kernel mean and covariance matrix.

        Returns:
            None

        Examples:
            >>> dataset = np.array([[0, 0, 0], [1, 1, 1]])
            >>> grf = GRF()
            >>> grf.assimilate_data(dataset)
            >>> grf.get_mu()
        """
        distance_min, ind_min_distance = self.grid_kdtree.query(dataset[:, :2])
        ind_assimilated = np.unique(ind_min_distance)
        salinity_assimilated = np.zeros([len(ind_assimilated), 1])
        for i in range(len(ind_assimilated)):
            ind_selected = np.where(ind_min_distance == ind_assimilated[i])[0]
            salinity_assimilated[i] = np.mean(dataset[ind_selected, -1])
        self.__update(ind_measured=ind_assimilated, salinity_measured=salinity_assimilated)

        self.__cnt += 1
        return ind_assimilated, salinity_assimilated, ind_min_distance

    def __update(self, ind_measured: np.ndarray, salinity_measured: np.ndarray) -> None:
        """
        Update GRF kernel based on sampled data.

        Args:
            ind_measured: indices where the data is assimilated.
            salinity_measured: measurements at sampeld locations, dimension: m x 1

        Methodology:
            1. Loop through each measurement and construct the measurement matrix F.
            2. Construct the measurement noise matrix R.
            3. Update the kernel mean and covariance matrix using
                .. math::

                    \mu = \mu + \Sigma F^T (F \Sigma F^T + R)^{-1} (y - F \mu)

                    \Sigma = \Sigma - \Sigma F^T (F \Sigma F^T + R)^{-1} F \Sigma

        Returns:
            None

        Examples:
            >>> ind_measured = np.array([0, 1])
            >>> salinity_measured = np.array([0, 1])
            >>> grf = GRF()
            >>> grf.__update(ind_measured, salinity_measured)
            >>> grf.get_mu()

        """
        msamples = salinity_measured.shape[0]
        F = np.zeros([msamples, self.grid.shape[0]])
        for i in range(msamples):
            F[i, ind_measured[i]] = True
        R = np.eye(msamples) * self.__tau ** 2
        C = F @ self.__Sigma @ F.T + R
        self.__mu = self.__mu + self.__Sigma @ F.T @ np.linalg.solve(C, (salinity_measured - F @ self.__mu))
        self.__Sigma = self.__Sigma - self.__Sigma @ F.T @ np.linalg.solve(C, F @ self.__Sigma)

    def get_eibv_at_locations(self, locations: np.ndarray) -> np.ndarray:
        """
        Calculate the EIBV for a given set of locations.

        Args:
            locations: np.array([x, y, z])

        Methodology:
            1. Get the indices of the locations.
            2. Calculate the EIBV using the analytical formula with fast approximation.
        """
        indices_candidates = self.get_ind_from_location(locations)
        eibv = np.zeros([len(indices_candidates), 1])
        for i in range(len(indices_candidates)):
            sigma_diag, vr_diag = self.__get_posterior_sigma_diag_vr_diag(indices_candidates[i])
            eibv[i] = self.__cal_analytical_eibv_fast(self.__mu, sigma_diag, vr_diag, self.__threshold)
            # eibv2[i] = self.__cal_analytical_eibv(self.__mu, sigma_diag, vr_diag, self.__threshold)
        return eibv

    def __get_posterior_sigma_diag_vr_diag(self, ind: int) -> tuple:
        """
        Calculate the diagonal of the posterior covariance matrix and the diagonal of the VR matrix.
        """
        F = np.zeros([1, self.__n_grf_grid])
        F[0, ind] = True
        R = np.eye(1) * self.__tau ** 2
        C = F @ self.__Sigma @ F.T + R
        VR = self.__Sigma @ F.T @ np.linalg.solve(C, F @ self.__Sigma)
        Sigma_posterior = self.__Sigma - VR
        sigma_diag = np.diag(Sigma_posterior)
        vr_diag = np.diag(VR)
        return sigma_diag, vr_diag

    def __cal_analytical_eibv_fast(self, mu: np.ndarray, sigma_diag: np.ndarray,
                                   vr_diag: np.ndarray, threshold: float) -> float:
        """
        Calculate the eibv using the analytical formula with a bivariate cumulative dentisty function.

        Args:
            mu: n x 1 dimension
            sigma_diag: n x 1 dimension
            vr_diag: n x 1 dimension

        Methodology:
            1. Calculate the probability of exceedance of the threshold using a bivariate cumulative dentisty function.
                .. math::
                    p = \\Phi(\\frac{\\theta - \mu}{\sigma}) - \\Phi(\\frac{\\theta - \mu}{\sigma}) \\Phi(\\frac{\\theta - \mu}{\sigma})

            2. Calculate eibv by summing up the product of p*(1-p).

        Returns:
            eibv: information based on variance reduction, dimension: n x 1

        Examples:
            >>> grf = GRF()
            >>> eibv = grf.__cal_analytical_eibv_fast(grf.__mu, grf.__sigma_diag, grf.__vr_diag)
        """
        # s1, calculate the z1, z2, rho
        mu = mu.squeeze()
        sn2 = sigma_diag
        vn2 = vr_diag
        sn = np.sqrt(sn2)
        mur = (np.ones_like(mu) * threshold - mu) / sn
        sig2r_1 = sn2 + vn2
        sig2r = vn2
        z1 = mur
        z2 = -mur
        rho = -sig2r / sig2r_1
        grid = np.stack((rho, z1, z2), axis=1)

        # s2, query the cdf from the loaded interpolators
        ebv = np.array([self.__query_cdf(rho, z1, z2) for rho, z1, z2 in grid])
        ebv[ebv < 0] = 0
        eibv = np.sum(ebv)
        return eibv

    def __cal_analytical_eibv(self, mu: np.ndarray, sigma_diag: np.ndarray,
                              vr_diag: np.ndarray, threshold: float) -> float:
        """
        Calculate the eibv using the analytical formula with a bivariate cumulative dentisty function.

        Input:
            - mu: mean of the posterior marginal distribution.
                - np.array([mu1, mu2, ...])
            - sigma_diag: diagonal elements of the posterior marginal variance.
                - np.array([sigma1, sigma2, ...])
            - vr_diag: diagonal elements of the variance reduction.
                - np.array([vr1, vr2, ...])
        """
        mu = mu.squeeze()
        eibv = .0
        EBV = []
        sn2 = sigma_diag
        vn2 = vr_diag
        sn = np.sqrt(sn2)

        mur = (threshold - mu) / sn

        sig2r_1 = sn2 + vn2
        sig2r = vn2

        z1 = mur
        z2 = -mur
        rho = -sig2r / sig2r_1
        for i in range(len(z1)):
            ebv = multivariate_normal.cdf([z1[i], z2[i]], mean=[0, 0], cov=[[1, rho[i]], [rho[i], 1]])
            eibv += ebv
            EBV.append(ebv)
        return eibv

    def __query_cdf(self, rho, z1, z2) -> np.ndarray:
        # s1, Find the index of the closest rho layer
        i = np.abs(self.__rho_values - rho).argmin()
        # s2, Use the interpolator for this layer to interpolate the value
        return self.__interpolators[i]([[z1, z2]])

    def get_ind_from_location(self, loc: np.ndarray) -> Union[int, np.ndarray, None]:
        """
        Args:
            loc: np.array([xp, yp, zp])
        Returns: index of the closest waypoint.
        """
        if len(loc) > 0:
            dm = loc.ndim
            if dm == 1:
                return self.grid_kdtree.query(loc.reshape(1, -1))[1]
            elif dm == 2:
                return self.grid_kdtree.query(loc)[1]
            else:
                return None
        else:
            return None

    def get_location_from_ind(self, ind: Union[int, list]) -> np.ndarray:
        """
        Args:
            ind: index of the locations.

        Returns: locations of the given indices.

        """
        return self.grid[ind, :]

    def set_threshold(self, value: float) -> None:
        """
        Set threshold.

        Args:
            value: threshold

        Examples:
            >>> grf = GRF()
            >>> grf.set_threshold(0.1)

        """
        self.__threshold = value

    def get_grid(self) -> np.ndarray:
        """
        Return grid of the GRF kernel.
        """
        return self.grid

    def get_threshold(self) -> float:
        """
        Return threshold.

        Returns:
            threshold: threshold

        Examples:
            >>> grf = GRF()
            >>> grf.get_threshold()
            27.0
        """
        return self.__threshold

    def get_mu(self) -> np.ndarray:
        """
        Return mean vector.

        Returns:
            mu: mean vector

        Examples:
            >>> grf = GRF()
            >>> grf.get_mu()
            array([0.1, 0.2, 0.3])

        """
        return self.__mu

    def get_mvar(self) -> np.ndarray:
        """
        Return marginal variance.
        """
        return np.diag(self.__Sigma)

    def interpolate_mu4locations(self, locations: np.ndarray) -> np.ndarray:
        """
        Interpolate mu at the given locations.
        """
        return self.__mu[self.grid_kdtree.query(locations)[1]]

    def get_covariance_matrix(self) -> np.ndarray:
        """
        Return Covariance.

        Returns:
            Sigma: Covariance matrix

        Examples:
            >>> grf = GRF()
            >>> grf.get_covariance_matrix()
            array([[1.00000000e+00, 9.99999998e-01, 9.99999994e-01],
                   [9.99999998e-01, 1.00000000e+00, 9.99999998e-01],
                   [9.99999994e-01, 9.99999998e-01, 1.00000000e+00]])

        """
        return self.__Sigma

    def get_random_realization(self) -> np.ndarray: 
        """
        Return a random realization of the GRF based on the prior mean and prior covariance matrix.
        
        Returns:
            realization: a random realization of the GRF

        Examples:
            >>> grf = GRF()
            >>> grf.get_random_realization()
            array([0.1, 0.2, 0.3])
        """
        realization = self.__mu_prior + np.linalg.cholesky(self.__Sigma_prior) @ np.random.randn(self.__n_grf_grid, 1)
        return realization

    def get_excursion_set(self) -> np.ndarray:
        """
        Return the excursion set based on the threshold.

        Returns:
            excursion_set: excursion set

        Examples:
            >>> grf = GRF()
            >>> grf.get_excursion_set()
            array([0.1, 0.2, 0.3])
        """
        excursion_set = np.zeros(self.__n_grf_grid)
        excursion_set[self.__mu.flatten() <= self.__threshold] = 1
        return excursion_set

    def get_excursion_probability(self) -> np.ndarray:
        """
        Return the excursion probability.

        Returns:
            excursion_probability: excursion probability

        Examples:
            >>> grf = GRF()
            >>> grf.get_excursion_probability()
            array([0.1, 0.2, 0.3])
        """
        return norm.cdf(self.__threshold, self.__mu.flatten(), np.sqrt(np.diag(self.__Sigma)))


if __name__ == "__main__":
    g = GRF()
