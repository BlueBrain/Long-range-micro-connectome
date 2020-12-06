import numpy


class RegionMapper(object):
    def __init__(self, cfg_file=None):
        if cfg_file is None:
            import os
            cfg_file = os.path.join(os.path.split(__file__)[0], 'default.json')
        if isinstance(cfg_file, dict):
            self.cfg = cfg_file
        else:
            from white_matter.utils.data_from_config import read_config
            self.cfg = read_config(cfg_file)["BrainParcellation"]
        self.region_names = list(map(str, self.cfg["region_names"]))
        self.module_names = list(map(str, self.cfg["module_names"]))
        self.source_names = list(map(str, self.cfg["projection_classes"]))
        self.module_idx = self.cfg["module_idx"]
        self.source_layers = self.cfg["class_to_layer"]
        self.source_filters = self.cfg["projection_class_fltrs"]

    def n_regions(self):
        return numpy.max(numpy.hstack(self.module_idx.values()))

    def idx2region(self, idx):
        if hasattr(idx, '__iter__'):
            return [self.region_names[i] for i in idx]
        return self.region_names[idx]

    def idx2module(self, idx):
        for k, v in self.module_idx.items():
            if v[0] <= idx < v[1]:
                return k
        return 'NONE'

    def region2idx(self, reg):
        return self.region_names.index(reg)

    def region2module(self, reg):
        return self.idx2module(self.region2idx(reg))

    def module2idx(self, module, is_not=False):
        idx = range(self.module_idx[module][0], self.module_idx[module][1])
        if is_not:
            return numpy.setdiff1d(range(self.n_regions()), idx)
        return idx

    def module2regions(self, module):
        idx = self.module2idx(module)
        return [self.region_names[i] for i in idx]
