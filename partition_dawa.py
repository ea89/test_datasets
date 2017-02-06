import numpy as np
from structuring import partition
import math 
from dpcomp_core.algorithm.dawa.partition_engines import l1partition
from op_transform import partition_misc as pm

"""
DAWA Partitioning
"""

def partition_dawa(x, prng, eps, ratio, approx=False):
    """
    Performs DAWA partitioning
    :param eps: real epsilon used by DAWA partition
    :return: Partition object
    """

    # get dawa clusters, default engine l1-approx as used in DAWA
    pseed = prng.randint(500000)

    if approx:
        cluster = l1partition.L1partition_approx(x, eps, ratio=ratio, gethist=True, seed=pseed)
    else:
        cluster = l1partition.L1partition(x, eps, ratio=ratio, gethist=True, seed=pseed)

    # convert cluster to partition object and set partition in state
    partition_vec = pm.get_partition_vec(None, len(x), cluster, closeRange= True)
    # return partition object
    return partition.Partition(partition_vec, canonical_order=False)



def hilbert_transform(shape1, shape2):
    '''
    Transform 2D domain to 1D domain according to hilbert curve. 
    Usesd in 2D DAWA as preprocessing. 
    The current implementation requires the domain to be a square of length which is a power of 2
    '''
    assert shape1 == shape2 and 2**int(math.ceil(math.log(shape1, 2))) == shape2

    x,y = hilbert(shape1)
    partition_vec = np.zeros_like(x)

    index = np.ravel_multi_index([x,y], (shape1,shape2))
    partition_vec[index] = range(len(x))

    return partition.Partition(partition_vec, canonical_order=False)



def hilbert(N):
    """
    Produce coordinates of an NxN Hilbert curve.

    @param N:
         the length of side, assumed to be a power of 2 ( >= 2)

    @returns:
          x and y, each as an array of integers representing coordinates
          of points along the Hilbert curve. Calling plot(x, y)
          will plot the Hilbert curve.

    From Wikipedia
    """
    assert 2**int(math.ceil(math.log(N, 2))) == N, "N={0} is not a power of 2!".format(N)
    if N==2:
        return  np.array((0, 0, 1, 1)), np.array((0, 1, 1, 0))
    else:
        x, y = hilbert(N/2)
        xl = np.r_[y, x,     N/2+x, N-1-y  ]
        yl = np.r_[x, N/2+y, N/2+y, N/2-1-x]
        return xl, yl

hilbert_transform(4,4)