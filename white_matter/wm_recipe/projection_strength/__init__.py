from ...utils.data_from_config import read_config as read_config_default
from ..parcellation import RegionMapper


def read_config(fn):
    from white_matter.utils.paths_in_config import path_local_to_cfg_root
    ret = read_config_default(fn)
    relevant_section = ret["ProjectionStrength"]
    relevant_section["cfg_root"] = ret["cfg_root"]
    path_local_to_cfg_root(relevant_section, ["cache_manifest", "h5_cache"])
    return relevant_section


class ProjectionStrength(object):

    def __init__(self, cfg_file=None):
        if cfg_file is None:
            import os
            cfg_file = os.path.join(os.path.split(__file__)[0], 'default.json')
            self.mpr = RegionMapper()
        else:
            self.mpr = RegionMapper(cfg_file=cfg_file)
        self.cfg_file = cfg_file
        self.cfg = read_config(cfg_file)
        self._DSET_SHAPE = (len(self.mpr.region_names), len(self.mpr.region_names))

    @staticmethod
    def _dict_to_path(D):
        return D.get("src_type", "wild_type") + '/' + D.get("hemi", "ipsi") + '/'\
               + D.get("measurement", "connection density")

    @staticmethod
    def layer_volume_fractions():
        import json, os
        fn = os.path.join(os.path.split(__file__)[0], 'relative_layer_volumes.json')
        with open(fn, 'r') as fid:
            ret = json.load(fid)
        return ret

    def _call_master(self):
        import h5py, os
        from .master_proj_mats import master_proj_mats
        res = master_proj_mats(self.cfg, self.mpr)
        if os.path.exists(self.cfg["h5_cache"]):
            h5 = h5py.File(self.cfg["h5_cache"], 'r+')
        else:
            h5 = h5py.File(self.cfg["h5_cache"], 'w')
        for k, v in res.items():
            h5.require_dataset(self._dict_to_path(dict(k)), self._DSET_SHAPE,
                               float, data=v)
        h5.close()

    def _normalized_per_layer(self, measurement):
        import numpy, h5py
        rel_vols = self.layer_volume_fractions()
        base_measurement = measurement[11:] #name of corresponding non-normalized measurement
        with h5py.File(self.cfg["h5_cache"], 'r+') as h5:
            for hemi in ["ipsi", "contra"]:
                B = self.__call__(measurement=base_measurement, src_type="wild_type", hemi=hemi)
                N = self.__call__(measurement=measurement, src_type="wild_type", hemi=hemi)
                V = numpy.vstack(numpy.mean(B / N, axis=1))
                for src_type in self.mpr.source_names:
                    tmp_type = src_type
                    fac = 1.0
                    if src_type.startswith('5'):
                        tmp_type = '5'
                        fac = 0.5 # Assumed split 50-50 between 5it and 5pt. Find a better solution in the future...
                    Vi = fac * V * numpy.vstack([rel_vols[_x][tmp_type] for _x in self.mpr.region_names])
                    M = self.__call__(measurement=base_measurement, src_type=src_type, hemi=hemi)
                    MN = M / Vi
                    MN[numpy.isinf(MN)] = numpy.NaN
                    print(self._dict_to_path({"measurement": measurement,
                                              "src_type": src_type, "hemi": hemi}))
                    h5.require_dataset(self._dict_to_path({"measurement": measurement,
                                                           "src_type": src_type, "hemi": hemi}),
                                       self._DSET_SHAPE, float, data=MN)
                    h5.flush()

    def _call_per_layer(self, measurement):
        if measurement.startswith("normalized"):
            self._normalized_per_layer(measurement)
            return
        import h5py
        from .per_layer_proj_mats import per_layer_proj_mats
        M_i = self.__call__(hemi="ipsi", src_type="wild_type", measurement=measurement)
        M_c = self.__call__(hemi="contra", src_type="wild_type", measurement=measurement)
        res = per_layer_proj_mats(self.cfg, self.mpr, M_i, M_c,
                                  scale=(measurement == "connection density"),
                                  vol_dict=self.layer_volume_fractions())
        with h5py.File(self.cfg["h5_cache"], 'r+') as h5:
            for k, v in res.items():
                D = dict(k)
                D["measurement"] = measurement
                h5.require_dataset(self._dict_to_path(D), self._DSET_SHAPE,
                                   float, data=v)
                h5.flush()

    def __call__(self, *args, **kwargs):
        measurement = kwargs.get("measurement", "connection density")
        src_type = str(kwargs.get("src_type", "wild_type"))
        import h5py, os, numpy
        if not os.path.exists(self.cfg["h5_cache"]):
            self._call_master()
        h5 = h5py.File(self.cfg["h5_cache"], "r")
        if self._dict_to_path(kwargs) not in h5:
            h5.close()
            if src_type == "wild_type":
                self._call_master()
            else:
                self._call_per_layer(measurement)
        h5 = h5py.File(self.cfg["h5_cache"], "r")
        if self._dict_to_path(kwargs) not in h5:
            h5.close()
            raise Exception("Unsupported combination of arguments: %s" % str(kwargs))
        return numpy.array(h5[self._dict_to_path(kwargs)])


