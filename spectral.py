# -*- coding:utf-8 -*-
"""
@author: LIU
@file:spectral.py
@time:2016/11/1 22:45
"""
import numpy as np
import math
import sys
from scipy.cluster.vq import kmeans2


def SpectralClustering(simi_matrix, cluster_num):
    N, N = np.shape(simi_matrix)
    DN = np.diag(1 / np.sqrt(np.sum(simi_matrix, axis=1)))
    LapN = np.eye(N) - np.dot(np.dot(DN, simi_matrix), DN)
    U, s, V = np.linalg.svd(LapN, full_matrices=True)
    kerN = U[:, N - cluster_num + 1:N]
    for i in range(N):
        kerN[i, :] = kerN[i, :] / np.linalg.norm(kerN[i, :])
    centroids, label = kmeans2(kerN, cluster_num, iter=20)
    del DN, U, s, V, kerN, centroids
    return label
