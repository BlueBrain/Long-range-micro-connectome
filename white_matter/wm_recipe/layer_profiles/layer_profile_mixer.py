from ..parcellation import RegionMapper
import numpy
from ...utils.data_from_config import ConfiguredDataSource, read_config

#mpr = RegionMapper()


class LayerProfiles(ConfiguredDataSource):
    relevant_section = "layer_profiles"
    relevant_chapter = "LayerProfiles"

    def __init__(self, cfg_file=None):
        if cfg_file is None:
            import os
            cfg_file = os.path.join(os.path.split(__file__)[0], 'default.json')
        super(LayerProfiles, self).__init__(cfg_file)
        self.N = self.cfg["layer_profile_number"]
        self.pattern_layers = self.cfg["layer_profile_layers"]
        self.parameterize(self.cfg)
        if self.cfg[self.__class__.relevant_section]["source"] == "digitize":
            for k, v in self.patterns.items():
                self.patterns[k] = v ** 2

    def __pattern_to_filenames__(self, pat):
        return dict([(i, pat % i) for i in range(1, self.N + 1)])


class SourceProfiles(ConfiguredDataSource):
    relevant_section = "frequency_per_source"
    relevant_chapter = "LayerProfiles"

    def __init__(self, cfg_file, mpr):
        super(SourceProfiles, self).__init__(cfg_file)
        self.groups = self.__groups__(mpr)
        self.parameterize(self.cfg)
        for k, v in self.patterns.items():
            self.patterns[k] = len(v) * v / v.sum()

    def __groups__(self, mpr):
        return mpr.source_names

    def __pattern_to_filenames__(self, pat):
        return dict([(grp, pat % grp) for grp in self.groups])


class ModuleProfiles(SourceProfiles):
    relevant_section = "frequency_per_module"

    def __groups__(self, mpr):
        return ['%s_intra' % k for k in mpr.module_names] \
             + ['%s_inter' % k for k in mpr.module_names]


class ProfileMixer(object):
    def __init__(self, proj_strength, cfg_file=None):
        if cfg_file is None:
            import os
            cfg_file = os.path.join(os.path.split(__file__)[0], 'default.json')
            self.mpr = RegionMapper()
        else:
            self.mpr = RegionMapper(cfg_file=cfg_file)
        self.cfg = read_config(cfg_file)["LayerProfiles"]
        self.profiles_s = SourceProfiles(cfg_file, self.mpr)
        self.profiles_m = ModuleProfiles(cfg_file, self.mpr)
        self.modules = self.mpr.module_idx
        self.pw_strength = proj_strength
        self._hierarchy = ['POL', 'VM', 'RE', 'PIL', 'PF',
                           'ACAd', 'AIv', 'TEa', 'ACAv', 'MOs', 'VISC', 'ORBvl',
                           'SSp-un', 'VISa', 'VISpor', 'VISam', 'AUDpo', 'VISpm',
                           'FRP', 'ORBl', 'PL',
                           'LP', 'PO', 'SMT', 'VAL', 'AV', 'LD', 'PT', 'PVT', 'CM',
                           'RSPagl', 'AId', 'ORBm', 'VISal',
                           'VISrl', 'ILA', 'SSp-tr', 'RSPd', 'MOp', 'VISli', 'VISl',
                           'RSPv', 'SSs', 'SSp-bfd', 'VISpl', 'SSp-m', 'SSp-ul', 'AIp',
                           'AUDd', 'SSp-ll', 'SSp-n', 'AUDp', 'VISp',
                           'AMd', 'AMv', "LGd-sh", "LGd-co", 'LGd-ip', 'VPM', 'VPMpc', 'CL',
                           "IMD", 'VPL', 'VPLpc', 'PCN', "MGd", "MGv", "MGm", "IAD"
                           ]  #TODO: Read from config
        # Core: always lowest in hier. matrix_m: always highest. The other two sometimes high, sometimes low.
        # For now placed in middle.
        self._m_hierarchy = ['t_matrix_m', 'prefrontal', 'anterolateral', 'medial',
                             't_matrix_f', 't_il', 'visual',
                             'temporal', 'somatomotor', 't_core']
        '''No hierarchy reported for 4 regions. We put them in the middle (index 30).'''
        self.idx2hierarchy = [self._hierarchy.index(_x) if _x in self._hierarchy
                              else 30 for _x in self.mpr.region_names]
        self.m_idx2hierarchy = [self._m_hierarchy.index(_x)
                                for _x in self.mpr.module_names]

    def predict_mix_from_sources(self, mod_fr, mod_to):
        kk = self.mpr.source_names
        idx_fr = self.mpr.module2idx(mod_fr)
        idx_to = self.mpr.module2idx(mod_to)
        pw_exists = [(self.pw_strength(src_type=_k)[:, idx_to][idx_fr] > 0)
                     for _k in kk]
        N = list(map(numpy.sum, pw_exists))
        N = numpy.array(N).astype(float) / numpy.sum(N)
        result = numpy.vstack([n * self.profiles_s.patterns[_k]
                               for _k, n in zip(kk, N)]).sum(axis=0)
        return result

    def projection_is_ff(self, i, j):
        m_i = self.mpr.idx2module(i)
        m_j = self.mpr.idx2module(j)
        if m_i == m_j or True:
            return self.idx2hierarchy[i] > self.idx2hierarchy[j]
        else:
            return self._m_hierarchy.index(m_i) > self._m_hierarchy.index(m_j)

    def mix_module(self, source, mod_fr, mod_to):
        if mod_fr == mod_to:
            mod = mod_fr + '_intra'
        else:
            mod = mod_fr + '_inter'
        adjust = self.profiles_m.patterns[mod] / self.predict_mix_from_sources(mod_fr, mod_to)
        return self.profiles_s.patterns[source] * adjust

    def mix(self, source, i, j, use_hierarchy=True):
        mod_fr = self.mpr.idx2module(i)
        mod_to = self.mpr.idx2module(j)
        result = self.mix_module(source, mod_fr, mod_to)
        if use_hierarchy:
            dir_idx = self.cfg.get("profile_directions", {  # Default value for backwards compatibility
                "feedforward_indices": [0, 2, 4],
                "feedback_indices": [1, 3, 5]
            })
            if self.projection_is_ff(i, j):
                result[dir_idx["feedback_indices"]] *= 0.5
            else:
                result[dir_idx["feedforward_indices"]] *= 0.5
        return result

    def max(self, source, i, j, **kwargs):
        rel = self.mix(source, i, j, **kwargs)
        return numpy.argmax(rel)

    def max_module(self, source, mod_fr, mod_to):
        rel = self.mix_module(source, mod_fr, mod_to)
        return numpy.argmax(rel)

    def full_mat(self, source, **kwargs):
        N = self.mpr.n_regions()
        return numpy.array([[self.max(source, i, j, **kwargs)
                             for j in range(N)] for i in range(N)])

    def compare_for_source(self, source):
        from matplotlib import pyplot as plt
        M = self.full_mat(source)
        H = numpy.histogram(M + 1, weights=(self.pw_strength(src_type=source) > 0).astype(float),
                            bins=range(1, 8))[0]
        H = 6 * H.astype(float) / H.sum()
        plt.figure()
        plt.plot(range(1, 7), H, label='model')
        plt.plot(range(1, 7), self.profiles_s.patterns[source], label='data')
        plt.title(source)
        plt.legend()

    def compare_for_module(self, module, inter):
        from matplotlib import pyplot as plt
        M = numpy.dstack([self.full_mat(_s) + 1
                          for _s in self.profiles_s.patterns.keys()])
        W = numpy.dstack([self.pw_strength(src_type=_s) > 0
                          for _s in self.profiles_s.patterns.keys()]).astype(float)
        M = M[self.modules[module][0]:self.modules[module][1], :, :]
        W = W[self.modules[module][0]:self.modules[module][1], :, :]
        if inter:
            suffix = '_inter'
            N = numpy.max(numpy.hstack([self.modules.values()]))
            idxx = numpy.setdiff1d(range(N), range(self.modules[module][0], self.modules[module][1]))
            M = M[:, idxx, :]; W = W[:, idxx, :]
        else:
            suffix = '_intra'
            M = M[:, self.modules[module][0]:self.modules[module][1], :]
            W = W[:, self.modules[module][0]:self.modules[module][1], :]
        H = numpy.histogram(M, weights=W, bins=range(1, 8))[0]
        H = 6 * H.astype(float) / H.sum()
        plt.figure()
        plt.plot(range(1, 7), H, label='model')
        plt.plot(range(1, 7), self.profiles_m.patterns[module + suffix], label='data')
        plt.title(module)
        plt.legend()

