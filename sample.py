#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
from scipy.spatial.distance import cdist, euclidean, mahalanobis
from mahalanobis_transformer import MahalanobisTransformer


# set print options
np.set_printoptions(
    precision=3,
    suppress=True,
    formatter={'float': '{:0.3f}'.format}
)
# Generate random variables
# which obey to the multivariate normal distribution
m = np.array([1, 2])
K = np.array([[2, 1], [1, 2]])
np.random.seed(seed=12)
X = np.random.multivariate_normal(mean=m, cov=K, size=3)
print("=== X (random variables of multivariate normal distribution) ===")
print(X)
# Mahalanobis Transform
transformer = MahalanobisTransformer().fit(X)
Z = transformer.transform(X)
print("=== Z (Mahalanobis transformed values of X) ===")
print(Z)
# Inverse Mahalanobis Transform
XX = transformer.inverse_transform(Z)
print("=== Inverse Mahalanobis transformed values of Z ===")
print(XX)
# Check Euclidean Distance of X
n_feature = X.shape[1]
print("=== Euclidean Distance of X (numpy...norm) ===")
print(
    np.array([
        np.linalg.norm(x, ord=2) for x in X
    ])
)
print("=== Euclidean Distance of X (scipy...euclidean) ===")
print(
    np.array([
        euclidean(x, np.zeros(n_feature)) for x in X
    ]).ravel()
)
print("=== Euclidean Distance of X (scipy...cdist) ===")
print(
    cdist(
        X,
        np.zeros((1, n_feature)),
        metric='euclidean'
    ).ravel()
)
# Check Mahalanobis Distance of X
vi = np.linalg.inv(np.cov(X, rowvar=False))
print("=== Mahalanobis Distance of X (scipy...mahalanobis) ===")
print(
    np.array([
        mahalanobis(x, X.mean(axis=0), VI=vi) for x in X
    ])
)
print("=== Mahalanobis Distance of X (scipy...cdist) ===")
print(
    cdist(
        X,
        X.mean(axis=0).reshape((1, n_feature)),
        metric='mahalanobis',
        VI=vi
    ).ravel()
)
# Check Euclidean Distance of Z
print("=== Euclidean Distance of Z (numpy...norm) ===")
print(
    np.array([np.linalg.norm(x, ord=2) for x in Z])
)
print("=== Euclidean Distance of Z (scipy...euclidean) ===")
print(
    np.array([
        euclidean(x, np.zeros(n_feature)) for x in Z
    ]).ravel()
)
print("=== Euclidean Distance of Z (scipy...cdist) ===")
print(
    cdist(Z, np.zeros((1, n_feature)), metric='euclidean').ravel()
)
