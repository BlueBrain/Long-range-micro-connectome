import numpy


class RegionMapper(object):
    def __init__(self):
        self.region_names = ['FRP','MOs','ACAd','ACAv','PL','ILA','ORBl','ORBm','ORBvl',
                             'AId','AIv','AIp','GU','VISC',
                             'SSs','SSp-bfd','SSp-tr','SSp-ll','SSp-ul','SSp-un','SSp-n','SSp-m','MOp',
                             'VISal','VISl','VISp','VISpl','VISli','VISpor','VISrl',
                             'VISa','VISam','VISpm','RSPagl','RSPd','RSPv',
                             'AUDd','AUDp','AUDpo','AUDv','TEa','PERI','ECT']
        self.module_names = ['prefrontal', 'anterolateral', 'somatomotor', 'visual', 'medial', 'temporal']
        self.source_names = ['23', '4', '5it', '5pt', '6']
        self.module_idx = {'prefrontal': [0, 9], 'anterolateral': [9, 14], 'somatomotor': [14, 23],
                           'visual': [23, 30], 'medial': [30, 36], 'temporal': [36, 43]}
        self.source_layers = {'23': ['l23'], '4': ['l4'], '5it': ['l5'],
                              '5pt': ['l5'], '6': ['l6a', 'l6b']}
        self.source_filters = {'23': {'synapse_type': 'EXC'}, '4': {'synapse_type': 'EXC'},
                               '5it': {'synapse_type': 'EXC', 'proj_type': 'intratelencephalic'},
                               '5pt': {'synapse_type': 'EXC', 'proj_type': 'pyramidal tract'},
                               '6': {'synapse_type': 'EXC'}}

    def n_regions(self):
        return numpy.max(numpy.hstack(self.module_idx.values()))

    def idx2region(self, idx):
        if hasattr(idx, '__iter__'):
            return [self.region_names[i] for i in idx]
        return self.region_names[idx]

    def idx2module(self, idx):
        for k, v in self.module_idx.items():
            if v[0] <= idx and v[1] > idx:
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
        idx = idx = self.module2idx(module)
        return [self.region_names[i] for i in idx]
