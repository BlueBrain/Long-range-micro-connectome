from white_matter.wm_recipe.region_mapper import RegionMapper
from white_matter.wm_recipe.sample_from_image import ImgSampler
import numpy


mpr = RegionMapper()


def read_config(fn):
    import json
    with open(fn, 'r') as fid:
        ret = json.load(fid)
    return ret["LayerProfiles"]


class LayerProfiles(object):
    def __init__(self, cfg_file=None):
        if cfg_file is None:
            import os
            cfg_file = os.path.join(os.path.split(__file__)[0], 'default.json')
        self.cfg = read_config(cfg_file)
        self.treat_config()
        self.parameterize(self.cfg)

    def parameterize(self, cfg):
        self.pat = cfg["layer_profile_filename"]
        self.N = cfg["layer_profile_number"]
        self.w = tuple(cfg["layer_profile_cbar_width"])
        self.h = cfg["layer_profile_cbar_height"]
        self.v = tuple(cfg["layer_profile_cbar_values"])
        self.shape = cfg["layer_profile_shape"]
        self.kwargs = cfg["layer_profile_cbar_kwargs"]
        self.kwargs["filename"] = cfg["layer_profile_cbar_filename"]
        self.pattern_layers = cfg["layer_profile_layers"]
        self.read()

    def treat_config(self):
        import os
        from white_matter.utils.paths_in_config import path_local_to_path
        suffix = "_filename"
        local_path = os.path.split(__file__)[0]
        lst_args = [str(k) for k in self.cfg.keys() if str(k).endswith(suffix)]
        path_local_to_path(self.cfg, local_path, lst_args)

    def read(self):
        self.patterns = {}
        for i in range(1, self.N + 1):
            I = ImgSampler(self.pat % i, cbar=(self.w, self.h, self.v), cbar_kwargs=self.kwargs)
            I.sample(*self.shape)
            I.map(lambda x: x ** 2)
            self.patterns[i] = I.out.copy()


class SourceProfiles(object):
    groups = mpr.source_names

    def __init__(self, cfg, prefix="rel_frequency"):
        self.pat = cfg[prefix + "_filename"]
        self.groups = self.__class__.groups
        self.w = tuple(cfg[prefix + "_cbar_width"])
        self.h = cfg[prefix + "_cbar_height"]
        self.v = cfg[prefix + "_cbar_values"]
        self.v = dict([(float(k), v) for k, v in self.v.items()])
        self.kwargs = cfg[prefix + "_cbar_kwargs"]
        self.kwargs["filename"] = cfg[prefix + "_cbar_filename"]
        self.sample_shape = tuple(cfg[prefix + "_shape"])
        self.read()

    def read(self):
        self.patterns = {}
        for grp in self.groups:
            I = ImgSampler(self.pat % grp, cbar=(self.w, self.h, self.v),
                           cbar_kwargs=self.kwargs)
            I.sample(*self.sample_shape)
            self.patterns[grp] = I.out[0]
            self.patterns[grp] = len(self.patterns[grp]) * self.patterns[grp] / self.patterns[grp].sum()


class ModuleProfiles(SourceProfiles):
    groups = ['%s_intra' % k for k in mpr.module_names] \
             + ['%s_inter' % k for k in mpr.module_names]

    def __init__(self, cfg):
        super(ModuleProfiles, self).__init__(cfg, prefix="module_frequency")

    def read(self):
        prop_order = [0, 3, 1, 4, 2, 5]
        super(ModuleProfiles, self).read()
        self.patterns = dict([(k, v[prop_order]) for k, v in self.patterns.items()])


class ProfileMixer(object):
    def __init__(self, proj_strength, cfg_file=None):
        if cfg_file is None:
            import os
            cfg_file = os.path.join(os.path.split(__file__)[0], 'default.json')
        self.cfg = read_config(cfg_file)
        self.treat_config()
        self.profiles_s = SourceProfiles(self.cfg)
        self.profiles_m = ModuleProfiles(self.cfg)
        self.modules = mpr.module_idx
        self.pw_strength = proj_strength
        self._hierarchy = ['ACAd', 'AIv', 'TEa', 'ACAv', 'MOs', 'VISC', 'ORBvl',
                          'SSp-un', 'VISa', 'VISpor', 'VISam', 'AUDpo', 'VISpm',
                          'FRP', 'ORBl', 'PL', 'RSPagl', 'AId', 'ORBm', 'VISal',
                          'VISrl', 'ILA', 'SSp-tr', 'RSPd', 'MOp', 'VISli', 'VISl',
                          'RSPv', 'SSs', 'SSp-bfd', 'VISpl', 'SSp-m', 'SSp-ul', 'AIp',
                          'AUDd', 'SSp-ll', 'SSp-n', 'AUDp', 'VISp'] #Harris et al, Fig8e
        self._m_hierarchy = ['prefrontal', 'anterolateral', 'medial', 'visual',
                             'temporal', 'somatomotor']
        '''No hierarchy reported for 4 regions. We put them in the middle (index 20).'''
        self.idx2hierarchy = [self._hierarchy.index(_x) if _x in self._hierarchy
                              else 20 for _x in mpr.region_names]
        self.m_idx2hierarchy = [self._m_hierarchy.index(_x)
                                for _x in mpr.module_names]

    @staticmethod
    def _treat_path(fn):
        import os
        if not os.path.isabs(fn):
            fn = os.path.join(os.path.split(__file__)[0], fn)
        return fn

    def treat_config(self):
        suffix = "_filename"
        for k, v in self.cfg.items():
            if str(k).endswith(suffix):
                self.cfg[k] = self._treat_path(v)

    def predict_mix_from_sources(self, mod_fr, mod_to):
        kk = mpr.source_names
        idx_fr = mpr.module2idx(mod_fr)
        idx_to = mpr.module2idx(mod_to)
        pw_exists = [(self.pw_strength(src_type=_k)[:, idx_to][idx_fr] > 0)
                     for _k in kk]
        N = map(numpy.sum, pw_exists)
        N = numpy.array(N).astype(float) / numpy.sum(N)
        result = numpy.vstack([n * self.profiles_s.patterns[_k]
                               for _k, n in zip(kk, N)]).sum(axis=0)
        return result


    def projection_is_ff(self, i, j):
        m_i = mpr.idx2module(i)
        m_j = mpr.idx2module(j)
        if m_i == m_j or True:
            return self.idx2hierarchy[i] > self.idx2hierarchy[j]
        else:
            return self._m_hierarchy.index(m_i) > self._m_hierarchy.index(m_j)

    def mix_module(self, source, mod_fr, mod_to):
        if mod_fr == mod_to:
            mod = mod_fr + '_intra'
        else:
            mod = mod_fr + '_inter'
        #return self.profiles_m.patterns[mod] * self.profiles_s.patterns[source]
        adjust = self.profiles_m.patterns[mod] / self.predict_mix_from_sources(mod_fr, mod_to)
        return self.profiles_s.patterns[source] * adjust

    def mix(self, source, i, j, use_hierarchy=True):
        mod_fr = mpr.idx2module(i)
        mod_to = mpr.idx2module(j)
        result = self.mix_module(source, mod_fr, mod_to)
        if use_hierarchy:
            if self.projection_is_ff(i, j):
                result[[1, 3, 5]] *= 0.5
            else:
                result[[0, 2, 4]] *= 0.5
        return result

    def max(self, source, i, j, **kwargs):
        rel = self.mix(source, i, j, **kwargs)
        return numpy.argmax(rel)

    def max_module(self, source, mod_fr, mod_to):
        rel = self.mix_module(source, mod_fr, mod_to)
        return numpy.argmax(rel)

    def full_mat(self, source, **kwargs):
        N = mpr.n_regions()
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

