import numpy
from .sample_from_image import ImgSampler


def read_config(fn):
    import json, os
    with open(fn, 'r') as fid:
        ret = json.load(fid)
    ret["cfg_root"] = os.path.split(fn)[0]
    return ret


class ConfiguredDataSource(object):
    relevant_chapter = '__not_existing__'
    relevant_section = "__not_existing__"

    def __init__(self, cfg_file):
        if isinstance(cfg_file, dict):
            self.cfg = cfg_file
            self.cfg_root = self.cfg.get("cfg_root", ".")
        else:
            self.cfg = read_config(cfg_file)
            self.cfg_root = self.cfg["cfg_root"]
            self.cfg = self.cfg[self.__class__.relevant_chapter]

    def parameterize(self, cfg):
        _rs = self.__class__.relevant_section
        if str(cfg[_rs]["source"]) == "digitize":
            self.digitize(cfg[_rs]["parameters"])
        elif str(cfg[_rs]["source"]) == "config":
            self.direct_read(cfg[_rs]["parameters"])
        else:
            raise Exception("Unknown data input method: %s" % str(cfg[_rs]["source"]))

    def treat_config(self, cfg):
        from white_matter.utils.paths_in_config import path_local_to_path
        suffix = "filename"
        lst_args = [str(k) for k in cfg.keys() if str(k).endswith(suffix)]
        path_local_to_path(cfg, self.cfg_root, lst_args)

    def __pattern_to_filenames__(self, pat):
        raise NotImplementedError()

    def digitize(self, cfg):
        self.treat_config(cfg)
        pat = cfg["filename"]
        w = cfg["cbar_width"]
        h = cfg["cbar_height"]
        v = cfg["cbar_values"]
        if isinstance(v, dict):
            v = dict([(float(k), v) for k, v in v.items()])
        else:
            v = tuple(v)
        shape = cfg["shape"]
        kwargs = cfg["cbar_kwargs"]
        kwargs["filename"] = cfg["cbar_filename"]
        self.patterns = {}
        filenames = self.__pattern_to_filenames__(pat)
        for i, _fn in filenames.items():
            I = ImgSampler(_fn, cbar=(w, h, v), cbar_kwargs=kwargs)
            I.sample(*shape)
            self.patterns[i] = I.out.copy()
            if self.patterns[i].shape[0] == 1:
                self.patterns[i] = self.patterns[i][0]

        if "reorder" in cfg:
            prop_order = cfg["reorder"]
            self.patterns = dict([(k, v[prop_order]) for k, v in self.patterns.items()])

    def direct_read(self, cfg):
        self.patterns = cfg["patterns"].copy()
        if cfg.get("keys", None) == "int":
            self.patterns = dict([(int(k), v) for k, v in self.patterns.items()])
        elif cfg.get("keys", None) == "float":
            self.patterns = dict([(float(k), v) for k, v in self.patterns.items()])
        elif cfg.get("keys", None) == "str":
            self.patterns = dict([(str(k), v) for k, v in self.patterns.items()])
        if cfg.get("values", None) == "array":
            self.patterns = dict([(k, numpy.array(v, dtype=float)) for k, v in self.patterns.items()])
        if "vals_nan" in cfg:
            for v in self.patterns.values():
                for _nan in cfg["vals_nan"]:
                    v[v == _nan] = numpy.NaN
        self.N = len(self.patterns)

    def condense(self, idx_fr, idx_to, func=numpy.nansum):
        z_fr = list(zip(idx_fr[:-1], idx_fr[1:]))
        z_to = list(zip(idx_to[:-1], idx_to[1:]))
        for k, v in self.patterns.items():
            self.patterns[k] = numpy.array([[func(v[fr[0]:fr[1], t[0]:t[1]])
                                             for t in z_to] for fr in z_fr])

