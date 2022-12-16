#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class MahalanobisTransformer(TransformerMixin, BaseEstimator):
    '''Constructs a transformer that transforms data
    so to the squared norm of the transformed data
    becomes Mahalanobis' distance.
    '''
    def __init__(
        self: MahalanobisTransformer,
        validate: bool = False
    ) -> None:
        self.ss = StandardScaler()
        self.mah = None
        return

    def _check_input(self: MahalanobisTransformer, X, reset: bool):
        return self._validate_data(
            X=X,
            y='no_validation',
            accept_sparce=False,
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
        normed = self.ss.fit_transform(X)
        self.mah = np.linalg.cholesky(
            np.linalg.inv(
                np.cov(normed, rowvar=False)
            )
        )
        return self

    def __sklearn_is_fitted__(self: MahalanobisTransformer) -> bool:
        '''Check the instance of transformer is fitted

        Returns
        -------
        is_fitted : bool
            if the instance has already fitted, returns True
        '''
        if self.mah is None:
            return False
        return True

    def transform(self: MahalanobisTransformer, X):
        '''Transform X to Mahalanobis Space

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
        '''fit and transform

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
        '''Transform X to Euclid Space

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.

        Returns
        -------
        X_out : array-like, shape (n_samples, n_features)
        '''
        X = self._check_input(X, reset=False)
        r = np.dot(np.linalg.inv(self.mah), X)
        return self.ss.inverse_transform(r)
