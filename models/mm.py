import numpy as np
import math
from scipy import linalg
import warnings

class LogMM:
    """ Log-Normal Mixture For one-dimensional data

    Parameters
    ----------
    n_components: int, defaults to 1.
        The number of mixture components

    tol: float, defaults to 1e-3
        The convergence threshold for EM

    max_iter: int, defaults to 100
        The maximum iteration threshold for EM

    weights_init: array-like, shape (n_components, ), optional
        PDF for components. [p1, p2, ... , pn]
        If not provided, it will be initalized evenly.

    means_init: array-like, shape (n_components, ), optional
        Initialization for Means. If not provided, calculate using kmeans.

    var_init: array-like, shape (n_components, ), optional
        Initialization for variance.
    """

    def __init__(self, n_components=1, tol=1e-3, max_iter=100, weights_init=None, means_init=None, var_init=None):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.weights_init = weights_init
        self.means_init = means_init
        self.var_init = var_init

    def _check_parameters(self):
        """Check the Log Mixture parameters are well defined."""
        if (self.weights_init is not None):
            if np.sum(weights_init) != 1:
                raise ValueError("Invalid value for initial weights: Weights array need to have sum equals to 1.")
            if len(weights_init) != self.n_components:
                raise ValueError("Invalid length for initial weights: Weights need to have the same length as number of components.")
        if (self.means_init is not None):
            if len(means_init) != self.n_components:
                raise ValueError("Invalid length for initial means: Means need to have the same length as number of components.")

    def initialize_means_and_vars(self, X):
        """Initialize means and vars based on sorted value
        Parameters
        ---------
        X: array-like, input
        """
        _sorted = sorted(X)
        mid = len(_sorted // 2)
        return (np.array([np.means(_sorted[:mid]), np.means(_sorted[mid:])]),
                np.array([np.var(_sorted[:mid]), np.var(_sorted[mid:])]))

    def _initialize(self, X):
        """Initialization of the Gaussian Mixture Parameters
        Parameters
        ----------
        X: array-like, input
        """
        means, var = initialize_means_and_vars(X)
        self._weights = (np.array([1/n_components for _ in n_components]) if
                self.weights_init is None else self.weights_init)
        self._means = (means if self.means_init is None
                else self.menas_init)
        self._vars = (var if self.var_init is None
                else self.var_init)

    def density_func(self, y, mean, var):
        """Calculation based on lognormal density function

        Parameters
        ----------
        y: float, data point
        mean, var: float, parameter for current model

        Return
        -------
        Calculated Result
        """
        std = var*(1/2)
        return (1/(y*std*np.sqrt(2*math.pi))) * np.exp(-(np.log(y)-mean)**2/(2*var))

    def _e_step(self, X):
        """The expectation step for EM algorithm
        Parameter
        ---------
        X: array-like, input

        It calculates the responsiveness of model k to data point yj

        Return
        --------
        res: matrix-like, has shape(len(X), n_components)
        """
        n_samples, _ = X.shape()
        res = np.zeros((n_samples, self.n_components))

        for idx, data_point in enumerate(X):
            denominator = np.sum([self._weights[j] * density_func(data_point, self._means[j], self._vars[j]) for j in range(self.n_components)])
            for j in range (self.n_components):
                numerator = self._weights[j] * density_func(data_point, self._means[j], self._vars[j])
                res[idx][j] = numerator / denominator

        return res

    def _m_step(self, X, res):
        """Expectation Maximizing For EM
        Parameters
        -----------
        X: Array-like, input
        res: Obtained from expectation step. Has shape (len(X), n_components). Responsiveness matrix.

        Return
        -------
        means: Updated means. Has shape (n_components,)
        vars: Updated vars, Has shape (n_components, )
        weights: Updated weights, Has shape (n_components, )
        """
        means = np.zeros((n_components, ))
        var = np.zeros((n_components, ))
        weights = np.zeros((n_components, ))

        n_samples, _ = X.shape()
        for i in range(n_components):
            means[i] = (np.sum([res[j][i] * X[j] for j in range(n_samples)])) / (np.sum(res, axis=0)[i])
            var[i] = (np.sum([res[j][i] * (np.log(X[j]) - self._means[i])**2])) / (np.sum(res, axis=0)[i])
            weights[i] = (np.sum(res, axis=0)[i]) / (n_samples)

        self._means = means
        self._vars = var
        self._weights = weights

    def calc_log_pdf_sum(self, X, resp_mat):
        n_samples, _ = X.shape()
        res = 0
        for k in range(n_components):
            std = np.sqrt(self._vars[k])
            for j in range(n_samples):
               res += (resp_mat[j][k]*(np.log(1/(X[j]*(2*math.pi)**(1/2)))-
                   np.log(std)-
                   ((np.log(X[j]) - self._means[k])**2/(2 * self._vars[k]))))
        return res
    def calculate_log_prob(self, X, resp_mat):
        resp_mult_weights = np.sum([np.sum(resp_mat, axis=0)[k] * np.log(self._weights[k]) for k in range(n_componenets)])
        log_pdf_sum =

    def initialize_resp(X):
        return np.tile(self._weights, (len(X), 1))

    def fit(self, X):
        """Estimate model parameters using X

        Parameters
        -----------
            X: array-like, shape (n_samples, 1)

        Returns
        --------
            None
        """

        # Initializing parameters
        self._initialize(X)
        resp_mat = initialize_resp(X)
        self.converged_ = False

        print("Start fitting the data to logNormal mixture model.")
        print("--------------------Parameters-------------------------")
        print(f"n_components = {self.n_components},tolerance={self.tol}, max_iter={self.max_iter}, init_means = {self._means}, init_variance = {self._vars}, init_weights = {self._weights}")
        print("-------------------------------------------------------")
        ### Function incomplete
        log_prob = self.calculate_log_prob(X, resp_mat)

        for n_iter in range(1, self.max_iter + 1):
            prev_log_prob = log_prob

            # Response matrix from e-step
            resp_mat = self._e_step(X)
            self._m_step(X, resp_mat)

            log_prob = calculate_log_prob(X, resp_mat)
            change = log_prob - prev_log_prob

            if abs(change) < self.tol:
                self.converged_ = True
                break

        if not self.converged_:
            warnings.warn('Initialization did not converge.'
                    'Try different init parameters, '
                    'or increase max_iter, tol '
                    'or check for degenerate data.')
        else:
            print("Successfully fit the data")
            print("-------Parameters--------")
            print(f"means: {self._means}, variance: {self._vars}, weights: {self._weights}")






