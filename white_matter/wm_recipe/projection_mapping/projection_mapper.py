import json, os
import h5py, numpy


def read_config(fn):
    from white_matter.utils.paths_in_config import path_local_to_cfg_root
    with open(fn, 'r') as fid:
        ret = json.load(fid)["ProjectionMapping"]
    ret["cfg_root"] = os.path.split(fn)[0]
    path_local_to_cfg_root(ret, ["cache_manifest", "h5_fn"])
    return ret


class ProjectionMapper(object):

    def __init__(self, cfg_file=None):
        if cfg_file is None:
            cfg_file = os.path.join(os.path.split(__file__)[0], 'default.json')
        self.cfg = read_config(cfg_file)
        if not os.path.exists(self.cfg["h5_fn"]):
            import subprocess, logging
            logging.getLogger(__file__).warning("Mapping cache does not exist at %s! Creating it now..." % self.cfg["h5_fn"])
            subprocess.check_call(["write_projection_mapping_cache.py", cfg_file])
            assert os.path.exists(self.cfg["h5_fn"]), "Mapping cache still missing!"

    def move_to_left_hemi(self, x):
        x_out = x.copy()
        pivot = 2 * self.cfg["hemi_mirror_at"]
        x_out[x >= self.cfg["hemi_mirror_at"]] = pivot - x_out[x >= self.cfg["hemi_mirror_at"]]
        return x_out

    def move_to_right_hemi(self, x):
        x_out = x.copy()
        pivot = 2 * self.cfg["hemi_mirror_at"]
        x_out[x < self.cfg["hemi_mirror_at"]] = pivot - x_out[x < self.cfg["hemi_mirror_at"]]
        return x_out

    def for_source(self, src):
        with h5py.File(self.cfg["h5_fn"], 'r') as h5:
            x = numpy.array(h5[src]['coordinates']['x'])
            y = numpy.array(h5[src]['coordinates']['y'])
            base_sys = h5[src]['coordinates'].attrs.get('base_coord_system', 'Allen Dorsal Flatmap')
        x = self.move_to_right_hemi(x)
        return x, y, base_sys

    def for_target(self, src):
        def _for_target(tgt, hemi):
            with h5py.File(self.cfg["h5_fn"], 'r') as h5:
                x = numpy.array(h5[src]['targets'][tgt]['coordinates/x'])
                y = numpy.array(h5[src]['targets'][tgt]['coordinates/y'])
                base_sys = h5[src]['targets'][tgt]['coordinates'].attrs.get('base_coord_system', 'Allen Dorsal Flatmap')
                var = h5[src]['targets'][tgt]['mapping_variance'][0]
            if hemi == 'ipsi':
                x = self.move_to_right_hemi(x)
            elif hemi == 'contra':
                x = self.move_to_left_hemi(x)
            return x, y, base_sys, var
        return _for_target
