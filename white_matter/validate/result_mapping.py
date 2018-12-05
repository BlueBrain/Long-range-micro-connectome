from white_matter.wm_recipe.projection_mapping import GeneralProjectionMapper
import numpy


class ProjectionResultBaryMapper(GeneralProjectionMapper):

    def __init__(self, annotation_volume, mapper, structure_tree, result):
        super(ProjectionResultBaryMapper, self).__init__(annotation_volume, mapper, structure_tree)
        self._result = result

    @classmethod
    def from_cache(cls, cache, result):
        from mcmodels.core.cortical_map import CorticalMap
        vol, _ = cache.get_annotation_volume()
        M = CorticalMap(projection='dorsal_flatmap')
        tree = cache.get_structure_tree()
        return cls(cache, vol, M, tree, result)

    def locs2vol(self, locs):
        vol = numpy.zeros(self._mapper.REFERENCE_SHAPE)
        xyz = numpy.round(locs).astype(int)
        for x, y, z in xyz:
            vol[x, y, z] += 1
        return vol

    def _for_full_volume(self, thresh=1.5, normalize=True):
        region = self._prepared_for
        import progressbar

        props = self._result._circ.v2.cells.get(group=self._result._pre_gids,
                                                properties=['region'])
        valid = props['region'] == region
        pre_gids = props.index.values[valid]
        locs = self._result._circ.v2.cells.get(pre_gids, properties=['x', 'y', 'z'])
        pts2d = self.transform_points(locs['x'].values, locs['y'].values, locs['z'].values)
        cols = self._bary.col(pts2d[1], pts2d[0])
        value = numpy.abs(numpy.diff(numpy.hstack([cols, cols[:, 0:1]]), axis=1)).sum(axis=1)
        valid = value > thresh
        cols = cols[valid]
        pre_gids = pre_gids[valid]
        post_locs = self._result.postsynaptic_locations(pre_gids, split=True)

        out_R = numpy.zeros(self._mapper.REFERENCE_SHAPE, dtype=float)
        out_G = numpy.zeros(self._mapper.REFERENCE_SHAPE, dtype=float)
        out_B = numpy.zeros(self._mapper.REFERENCE_SHAPE, dtype=float)
        out_count = numpy.zeros(self._mapper.REFERENCE_SHAPE, dtype=float)

        pbar = progressbar.ProgressBar()
        for _col, _locs in pbar(zip(cols, post_locs)):
            proj = self.locs2vol(_locs)
            out_count += proj
            out_R += _col[0] * proj
            out_G += _col[1] * proj
            out_B += _col[2] * proj

        if normalize:
            out_R /= out_count; out_G /= out_count; out_B /= out_count
        return out_R, out_G, out_B
