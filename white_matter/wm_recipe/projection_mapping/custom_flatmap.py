import numpy
import voxcell


class NrrdFlatmap(object):
    def __init__(self, fn):
        self.raw_map = voxcell.VoxelData.load_nrrd(fn)
        self._something()

    def transform(self, vol3d, agg_func=numpy.nansum):
        assert vol3d.shape == self.REFERENCE_SHAPE
        out = numpy.zeros(self.view_lookup.shape, dtype=vol3d.dtype)  # Create output array
        nz = numpy.nonzero(self.view_lookup > -1)  # For every pixels that has something mapped to it
        for i, j in zip(*nz):
            flat_idx = self.paths[self.view_lookup[i, j]]  # Look up the path for that pixel
            flat_idx = flat_idx[flat_idx > 0] - 1  # Only entries larger than zero
            out[i, j] = agg_func(vol3d.flat[flat_idx])
        return out

    def _something(self):
        raw = self.raw_map.raw

        vol = numpy.any(raw > -1, axis=-1)
        self.REFERENCE_SHAPE = vol.shape
        nz = numpy.nonzero(vol)
        idx = list(map(tuple, raw[nz]))  # List of flat coordinates of voxels that are mapped
        view_shape = tuple(raw[nz].max(axis=0) + 1)  # Size of flattened coordinate system
        nz = numpy.nonzero(vol.reshape(-1))[0]  # flat indices of voxels that are mapped

        path_map = {}
        self.view_lookup = -numpy.ones(view_shape, dtype=int)  # lookup in flat coordinates
        for tpl, flat_index in zip(idx, nz):  # For all voxels that are mapped
            if tpl in path_map:  # If a voxel at the same flat coordinate has already been mapped
                path_idx, path_list = path_map[tpl]  # Look it up in path_map
            else:  # else create new entry
                path_idx = len(path_map)  # path_index gets a new unique index
                path_list = []  # path_list begins empty...
                path_map[tpl] = (path_idx, path_list)  # register with path_map
                self.view_lookup[tpl[0], tpl[1]] = path_idx  # and register in view_lookup
            path_list.append(flat_index + 1)  # finally, add flat index to the new or existing path_list
        max_path_len = numpy.max([len(x[1]) for x in path_map.values()])
        self.paths = numpy.zeros((len(path_map), max_path_len), dtype=int)
        for path_num, path_list in path_map.values():
            self.paths[path_num, :len(path_list)] = path_list


