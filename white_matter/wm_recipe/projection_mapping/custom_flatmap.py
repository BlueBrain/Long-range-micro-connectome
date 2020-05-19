import numpy
import voxcell


class NrrdFlatmap(object):
    def __init__(self, fn):
        self.raw_map = voxcell.VoxelData.load_nrrd(fn)
        self._something()

    def transform(self, vol3d, agg_func=numpy.nansum):
        assert vol3d.shape == self.REFERENCE_SHAPE
        out = numpy.zeros(self.view_lookup.shape, dtype=vol3d.dtype)
        nz = numpy.nonzero(self.view_lookup > -1)
        for i, j in zip(*nz):
            flat_idx = self.paths[self.view_lookup[i, j]]
            flat_idx = flat_idx[flat_idx > 0] - 1
            out[i, j] = agg_func(vol3d.flat[flat_idx])
        return out

    def _something(self):
        raw = self.raw_map.raw

        vol = numpy.any(raw > -1, axis=-1)
        self.REFERENCE_SHAPE = vol.shape
        nz = numpy.nonzero(vol)
        idx = map(tuple, raw[nz])
        view_shape = tuple(raw[nz].max(axis=0) + 1)
        nz = numpy.nonzero(vol.reshape(-1))[0] # because a 0 in paths means no value..?

        path_map = {}
        self.view_lookup = -numpy.ones(view_shape, dtype=int)
        for tpl, flat_index in zip(idx, nz):
            if tpl in path_map:
                path_idx, path_list = path_map[tpl]
            else:
                path_idx = len(path_map)
                path_list = []
                path_map[tpl] = (path_idx, path_list)
                self.view_lookup[tpl[0], tpl[1]] = path_idx
            path_list.append(flat_index)
        max_path_len = numpy.max([len(x[1]) for x in path_map.values()])
        self.paths = numpy.zeros((len(path_map), max_path_len), dtype=int)
        for path_num, path_list in path_map.values():
            self.paths[path_num, :len(path_list)] = path_list


