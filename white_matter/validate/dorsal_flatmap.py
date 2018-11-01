from mcmodels.core.cortical_map import CorticalMap
from mcmodels.core import VoxelModelCache
from white_matter.wm_recipe import region_mapper
import numpy


class DorsalFlatmap(object):
    def __init__(self, manifest_file=None):
        self._mapper = CorticalMap(projection='dorsal_flatmap')
        if manifest_file is None:
            import os
            manifest_file = os.path.join(os.getenv('HOME', '.'), 'data/mcmodels')
            if not os.path.exists(manifest_file):
                os.makedirs(manifest_file)
            manifest_file = os.path.join(manifest_file, 'cache.json')
        self._cache = VoxelModelCache(manifest_file=manifest_file)
        self._vol = self._cache.get_annotation_volume()[0]
        self._tree = self._cache.get_structure_tree()
        self._mpr = region_mapper.RegionMapper()
        self._make_reverse_lookup()
        self._make_mapped_idx()

    def _three_d2flat_idx(self, idx):
        return idx[:, 0] * numpy.prod(self._mapper.REFERENCE_SHAPE[1:]) + \
               idx[:, 1] * self._mapper.REFERENCE_SHAPE[-1] + idx[:, 2] + 1

    def _coordinates2voxel(self, xyz, cutoff=4):
        xyz = (xyz / 100).astype(int)
        res = []
        for pt in xyz:
            square_d = numpy.sum((self._mapped_idx - pt) ** 2, axis=1)
            idx = numpy.argmin(square_d)
            if square_d[idx] <= cutoff:
                res.append(self._mapped_idx[idx])
            else:
                res.append([0, 0, 0])
        return numpy.vstack(res)


    def _make_reverse_lookup(self):
        Y, X = numpy.meshgrid(xrange(self._mapper.view_lookup.shape[1]),
                              xrange(self._mapper.view_lookup.shape[0]))
        self._rv_lookup = dict([(idx, (x, y)) for x, y, idx in
                                zip(X.flat, Y.flat, self._mapper.view_lookup.flat)])

    def _make_mapped_idx(self):
        is_mapped = numpy.zeros(self._mapper.REFERENCE_SHAPE, dtype=bool)
        is_mapped.flat[self._mapper.paths[self._mapper.paths > 0] - 1] = True
        self._mapped_idx = numpy.vstack(numpy.nonzero(is_mapped)).transpose()

    def _region_ids(self, regions, resolve_to_leaf=False):
        if not isinstance(regions, list) or isinstance(regions, numpy.ndarray):
            regions = [regions]
        r_struc = self._tree.get_structures_by_acronym(regions)
        r_ids = numpy.array([_x['id'] for _x in r_struc])
        def resolver(r_ids):
            rslvd = [resolver(_chldr) if len(_chldr) else _base
                     for _base, _chldr in
                     zip(r_ids, self._tree.child_ids(r_ids))]
            return numpy.hstack(rslvd)

        if resolve_to_leaf:
            return resolver(r_ids)
        return r_ids

    def _make_volume_mask(self, idxx):
        return numpy.in1d(self._vol.flat, idxx).reshape(self._vol.shape)

    def make_3d_mask(self, regions):
        idxx = self._region_ids(regions, resolve_to_leaf=True)
        return self._make_volume_mask(idxx)

    def make_2d_mask(self, regions):
        return self._mapper.transform(self.make_3d_mask(regions))

    def transform_points(self, x, y, z):
        xyz = self._coordinates2voxel(numpy.vstack([x, y, z]).transpose())
        flt = self._three_d2flat_idx(xyz)
        flt2idx = {}
        for i, v in enumerate(flt):
            flt2idx.setdefault(v, []).append(i)
        hits = numpy.in1d(self._mapper.paths.flat, flt).reshape(self._mapper.paths.shape)
        idxx = numpy.nonzero(hits.sum(axis=1))[0]
        out_coords = numpy.NaN * numpy.ones((xyz.shape[0], 2))
        for idx in idxx:
            nz = self._rv_lookup[idx]
            src_flts = numpy.intersect1d(self._mapper.paths[idx], flt)
            for _flt in src_flts:
                out_coords[flt2idx[_flt], :] = [nz[0].mean(), nz[1].mean()]
        return out_coords

    def draw_modules(self, ax, color='black'):
        for mdl in self._mpr.module_names:
            ax.contour(self.make_2d_mask(self._mpr.module2regions(mdl)),
                       levels=[0.5], colors=[color], linewidths=1.5)

    def draw_region(self, ax, region, color='red'):
        ax.contour(self.make_2d_mask([region]),
                   levels=[0.5], colors=[color], linewidths=0.5)
