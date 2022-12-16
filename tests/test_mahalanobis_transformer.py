#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Callable
import pytest  # noqa: F401
import numpy as np
from scipy.spatial.distance import cdist
from mahalanobis_transformer import MahalanobisTransformer


class TestMahalanobisTransformer:
    def setup_method(
        self: TestMahalanobisTransformer,
        method: Callable
    ) -> None:
        '''
        入力となる乱数を生成する
        '''
        m = np.array([1., 2.])
        K = np.array([[2., 1.], [1., 2.]])
        self.X = np.random.multivariate_normal(mean=m, cov=K, size=1000)
        return

    def test_is_fitted(self: TestMahalanobisTransformer) -> None:
        '''
        __sklearn_is_fitted__ のテスト
        '''
        transformer = MahalanobisTransformer()
        assert(transformer.__sklearn_is_fitted__() is False)
        transformer.fit(self.X)
        assert(transformer.__sklearn_is_fitted__() is True)
        return

    def test_transfer_inverse(self: TestMahalanobisTransformer) -> None:
        '''
        transform()してinverse_transform()すると元に戻るか確認する
        '''
        transformer = MahalanobisTransformer().fit(self.X)
        Z = transformer.transform(self.X)
        np.testing.assert_equal(self.X.shape, Z.shape)
        XX = transformer.inverse_transform(Z)
        np.testing.assert_equal(self.X.shape, XX.shape)
        np.testing.assert_almost_equal(self.X, XX)
        return

    def test_distance(self: TestMahalanobisTransformer) -> None:
        '''
        変換した値のユークリッド距離が元のマハラノビス距離と同じか確認する
        '''
        n_feature = self.X.shape[1]
        vi = np.linalg.inv(np.cov(self.X, rowvar=False))
        mah = cdist(
            self.X,
            self.X.mean(axis=0).reshape((1, n_feature)),
            metric='mahalanobis',
            VI=vi
        ).ravel()
        transformer = MahalanobisTransformer().fit(self.X)
        Z = transformer.transform(self.X)
        euc = cdist(
            Z,
            np.zeros((1, n_feature)),
            metric='euclidean'
        ).ravel()
        np.testing.assert_equal(mah.shape, euc.shape)
        np.testing.assert_almost_equal(mah, euc)
        return

    def test_distance2(self: TestMahalanobisTransformer) -> None:
        '''
        変換した値のユークリッド距離が元のマハラノビス距離と同じか確認する
        （fit_transform() を使った場合）
        '''
        n_feature = self.X.shape[1]
        vi = np.linalg.inv(np.cov(self.X, rowvar=False))
        mah = cdist(
            self.X,
            self.X.mean(axis=0).reshape((1, n_feature)),
            metric='mahalanobis',
            VI=vi
        ).ravel()
        Z = MahalanobisTransformer().fit_transform(self.X)
        euc = cdist(
            Z,
            np.zeros((1, n_feature)),
            metric='euclidean'
        ).ravel()
        np.testing.assert_equal(mah.shape, euc.shape)
        np.testing.assert_almost_equal(mah, euc)
        return
