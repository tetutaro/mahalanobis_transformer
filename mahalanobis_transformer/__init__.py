#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class MahalanobisTransformer(TransformerMixin, BaseEstimator):
    '''Constructs a transformer that transforms data
    so to squared norm of transformed data
    becomes Mahalanobis' distance.

    Examples
    --------
    >>> import numpy as np
    >>> from mahalanobis_transformer import MahalanobisTransformer
    >>> m = np.array([1, 2])
    >>> K = np.array([[2, 1], [1, 2]])
    >>> np.random.seed(seed=12)
    >>> X = np.random.multivariate_normal(mean=m, cov=K, size=3)
    >>> X
    array([[ 0.903,  0.939]
           [ 1.906,  0.500]
           [ 1.163, -0.008]])
    >>> transformer = MahalamobisTransformer().fit(X)
    >>> Z = transformer.transform(X)
    >>> Z
    array([[-0.619,  0.975]
           [ 1.154,  0.049]
           [-0.534, -1.024]])
    >>> transformer.inverse_transform(Z)
    array([[ 0.903,  0.939]
           [ 1.906,  0.500]
           [ 1.163, -0.008]])
    >>> from sklearn.preprocessing import StandardScaler
    >>> from scipy.spatial.distance import cdist
    >>> ss = StandardScaler().fit(X)
    >>> X_normed = ss.transform(X)
    >>> vi = np.linalg.inv(np.cov(X_normed, rowvar=False))
    >>> cdist(X_normed, np.zeros((1, 2)), metric='mahalanobis', VI=vi).ravel()
    array([1.155, 1.155, 1.155])
    >>> cdist(Z, metric='euclidean').ravel()
    array([1.155, 1.155, 1.155])
    '''
    def __init__(
        self: MahalanobisTransformer,
    ) -> None:
        self.ss = None
        self.mah = None
        return

    def _check_input(self: MahalanobisTransformer, X, reset: bool):
        return self._validate_data(
            X=X,
            y='no_validation',
            accept_sparse=False,
            reset=reset
        )

    def fit(
        self: MahalanobisTransformer,
        X,
        y=None
    ) -> MahalanobisTransformer:
        '''Fit transformer by checking X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            MahalanobisTransformer class instance.
        '''
        X = self._check_input(X, reset=True)
        n_feature = X.shape[1]
        self.ss = StandardScaler().fit(X)
        normed = self.ss.transform(X)
        np.testing.assert_almost_equal(
            normed.mean(axis=0),
            np.zeros(n_feature)
        )
        np.testing.assert_almost_equal(
            normed.var(axis=0),
            np.ones(n_feature)
        )
        self.mah = np.linalg.cholesky(
            np.linalg.inv(
                np.cov(normed, rowvar=False)
            )
        )
        return self

    def __sklearn_is_fitted__(self: MahalanobisTransformer) -> bool:
        '''Check the instance of transformer has already fitted.

        Returns
        -------
        is_fitted : bool
            if the instance has already fitted, returns True.
        '''
        if self.mah is None:
            return False
        return True

    def transform(self: MahalanobisTransformer, X):
        '''Mahalanobis transform X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.

        Returns
        -------
        X_out : array-like, shape (n_samples, n_features)
            Transformed input.
        '''
        X = self._check_input(X, reset=False)
        normed = self.ss.transform(X)
        return np.dot(normed, self.mah)

    def fit_transform(self: MahalanobisTransformer, X):
        '''Fit to data, then transform it.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.

        Returns
        -------
        X_out : array-like, shape (n_samples, n_features)
            Transformed input.
        '''
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self: MahalanobisTransformer, X):
        '''Inverse Mahalanobis transform X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.

        Returns
        -------
        X_out : array-like, shape (n_samples, n_features)
            Inverse transformed input.
        '''
        X = self._check_input(X, reset=False)
        X_out = np.dot(X, np.linalg.inv(self.mah))
        return self.ss.inverse_transform(X_out)
