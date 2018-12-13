from white_matter.wm_recipe.projection_mapping import GeneralProjectionMapper
import numpy
import logging


log = logging.getLogger("result_mapping.py")


class ProjectionResultBaryMapper(GeneralProjectionMapper):

    def __init__(self, annotation_volume, mapper, structure_tree, result, proj_root):
        super(ProjectionResultBaryMapper, self).__init__(annotation_volume, mapper, structure_tree)
        self._result = result
        self._proj_root = proj_root

    @classmethod
    def from_cache(cls, cache, result, projection_root=None):
        from mcmodels.core.cortical_map import CorticalMap
        vol, _ = cache.get_annotation_volume()
        M = CorticalMap(projection='dorsal_flatmap')
        tree = cache.get_structure_tree()
        return cls(vol, M, tree, result, projection_root)

    def locs2vol(self, locs):
        vol = numpy.zeros(self._mapper.REFERENCE_SHAPE)
        xyz = numpy.round(locs.values / 100).astype(int)
        for x, y, z in xyz:
            vol[x, y, z] += 1
        return vol

    def transform_points_v2(self, pts_frame):
        if self._proj_root is None:
            raise Exception("transform_points_v2 only works when the location of reverse lookup files has been provided")
        xyz = (pts_frame / 100).astype(int)
        u_xyz = xyz.drop_duplicates()
        out = numpy.NaN * numpy.ones((len(pts_frame), 2))
        import os, h5py, pandas
        fn = os.path.join(self._proj_root, 'flatmap_rv_projection_%s.h5' % self._prepared_for)
        h5 = h5py.File(fn, 'r')
        for _, _xyz in u_xyz.iterrows():
            dset = 'x_%d/y_%d/z_%d' % tuple(_xyz)
            if dset in h5:
                frame = pandas.read_hdf(fn, dset)
                loc2d = (numpy.vstack(frame.index.values) * frame.values).sum(axis=0)
                idxx = (xyz == _xyz).all(axis=1)
                out[idxx.values, :] = loc2d
        return out

    def _for_full_volume(self, thresh=1.5, normalize=True):
        region = self._prepared_for
        import progressbar

        log.info("Finding presynaptic projecting neurons in %s" % region)
        props = self._result._circ.v2.cells.get(group=self._result._pre_gids,
                                                properties='region')
        valid = numpy.in1d(props, [region + str(i) for i in range(1, 7)])
        pre_gids = props.index.values[valid]

        log.info("Transforming their locations into flatmap 2d")
        locs = self._result._circ.v2.cells.get(pre_gids, properties=['x', 'y', 'z'])

        #Hack: limit to one hemisphere
        pre_gids = locs[locs['z'] > 5750].index.values
        locs = locs[locs['z'] > 5750]
        #end of hack

        pts2d = self.transform_points_v2(locs)
        valid = ~numpy.any(numpy.isnan(pts2d), axis=1)
        pre_gids = pre_gids[valid]
        pts2d = pts2d[valid]

        log.info("Getting colors associated with their 2d locations")
        cols = self._bary.col(pts2d[:, 1], pts2d[:, 0])
        value = numpy.abs(numpy.diff(numpy.hstack([cols, cols[:, 0:1]]), axis=1)).sum(axis=1)
        valid = value > thresh

        log.info("Getting postsynaptic locations of their projection synapses")
        cols = cols[valid]
        pre_gids = pre_gids[valid]
        #post_locs = self._result.postsynaptic_locations(pre_gids, split=True)
        post_locs = self._result.postsynaptic_locations(pre_gids, split=False)
        S = self._result._postsynaptic_property(pre_gids, 'sgid')

        out_R = numpy.zeros(self._mapper.REFERENCE_SHAPE, dtype=float)
        out_G = numpy.zeros(self._mapper.REFERENCE_SHAPE, dtype=float)
        out_B = numpy.zeros(self._mapper.REFERENCE_SHAPE, dtype=float)
        out_count = numpy.zeros(self._mapper.REFERENCE_SHAPE, dtype=float)

        log.info("Generating 3d volume of postsynaptic locations")
        pbar = progressbar.ProgressBar()
        for _col, _gid in pbar(zip(cols, pre_gids)):
            proj = self.locs2vol(post_locs[S == _gid])
            out_count += proj
            out_R += _col[0] * proj
            out_G += _col[1] * proj
            out_B += _col[2] * proj

        if normalize:
            out_R /= out_count; out_G /= out_count; out_B /= out_count
        return out_R, out_G, out_B
