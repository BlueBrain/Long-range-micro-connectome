#!/usr/bin/env python
import numpy
import multiprocessing
from white_matter.validate import DorsalFlatmap
import logging
import pandas


logging.basicConfig(level=1)
__log__ = logging.getLogger(__file__)


class IdealizedPath(object):
    @classmethod
    def from_data(cls, p_xyz):
        p_xyz = p_xyz[~numpy.any(numpy.isnan(p_xyz), axis=1)]
        if p_xyz.shape[0] == 0:
            return None
        elif p_xyz.shape[0] == 1:
            return DegeneratePath(p_xyz[0])
        return LinearFittedPath(p_xyz)


class DegeneratePath(IdealizedPath):

    def __init__(self, p_xyz):
        self.p_xyz = p_xyz

    def len_intersection(self, vx, vy, vz):
        if numpy.sum((numpy.array([vx, vy, vz]) - self.p_xyz) ** 2) >= 0.25:
            return 0.0
        return numpy.sqrt(3)


class LinearFittedPath(IdealizedPath):

    def __init__(self, p_xyz):
        fx, ox = numpy.polyfit(range(p_xyz.shape[0]), p_xyz[:, 0], 1)
        fy, oy = numpy.polyfit(range(p_xyz.shape[0]), p_xyz[:, 1], 1)
        fz, oz = numpy.polyfit(range(p_xyz.shape[0]), p_xyz[:, 2], 1)
        self._offset = numpy.array([ox, oy, oz]) #l0
        self._l = numpy.array([fx, fy, fz]) #l
        self._params = (fx, ox, fy, oy, fz, oz)

    def intersect_plane(self, p0, n):
        paralell = numpy.dot(self._l, n)
        d = numpy.dot(p0 - self._offset, n) / paralell
        if paralell == 0:
            if d != 0:
                return None
            raise Exception("Handle this case if it ever occurs!")
        return self._offset + d * self._l

    def constrained_intersect_plane(self, p0, n):
        P = self.intersect_plane(p0, n)
        if P is None:
            return P
        for _n, _p, _p0 in zip(n, P, p0):
            if (_n == 0) and (numpy.abs(_p - _p0) > 0.5):
                return None
        return P

    def gen_planes(self, vx, vy, vz):
        p_base = numpy.array([vx, vy, vz], dtype=float)
        for i in range(3):
            n = numpy.zeros(3, dtype=float)
            n[i] = 1.0
            for a in [-0.5, 0.5]:
                p0 = p_base + a * n
                yield (p0, n)

    def intersect_voxel(self, vx, vy, vz):
        intersections = [self.constrained_intersect_plane(*p)
                         for p in self.gen_planes(vx, vy, vz)]
        return [p for p in intersections if p is not None]

    def len_intersection(self, vx, vy, vz):
        P = self.intersect_voxel(vx, vy, vz)
        if len(P) == 2:
            return numpy.sqrt(numpy.sum((P[1] - P[0]) ** 2))
        return 0.0


D = DorsalFlatmap()
paths = [IdealizedPath.from_data(D._flat_idx2three_d(_path))
         for _path in D._mapper.paths]

def weights_for_voxel(loc):
    global D
    global paths
    img_shape = D._mapper.view_lookup.shape
    __log__.info(str(loc))
    vx, vy, vz = loc
    W = [_path.len_intersection(vx, vy, vz) for _path in paths]
    if not numpy.any(W):
        index = pandas.MultiIndex(levels=(range(img_shape[0]), range(img_shape[1])),
                                  labels=[numpy.empty((0,), dtype=int), numpy.empty((0,), dtype=int)],
                                  names=('x', 'y'))
        return pandas.DataFrame(data=numpy.empty((0,), dtype=float), index=index, columns=['weight'])
    xy_index, weight = zip(*[(D._rv_lookup[i], v) for i, v in enumerate(W) if v > 0])
    xy_labels = map(list, numpy.vstack(xy_index).transpose())
    weight = numpy.array(weight) / numpy.sum(weight)
    index = pandas.MultiIndex(levels=(range(img_shape[0]), range(img_shape[1])),
                              labels=xy_labels, names=('x', 'y'))
    return pandas.DataFrame(data=numpy.array(weight), index=index, columns=['weight'])


if __name__ == "__main__":
    import sys
    fn_out = sys.argv[1]
    xfrom, xto, yfrom, yto, zfrom, zto = map(int, sys.argv[2:])
    X, Y, Z = numpy.meshgrid(range(xfrom, xto), range(25, 35), range(40, 50))
    P = multiprocessing.Pool(128)
    res = P.map(weights_for_voxel, zip(X.flat, Y.flat, Z.flat))
    for x, y, z, data in zip(X.flat, Y.flat, Z.flat, res):
        data.to_hdf(fn_out, 'x_%d/y_%d/z_%d' % (x, y, z))
