import numpy


class LayerProfileVsRawData(object):

    def __init__(self, cache, mpr, mxr, layer_profiles):
        self._cache = cache
        self._mpr = mpr
        self._mxr = mxr
        self._profiles = layer_profiles
        self._vol = self._cache.get_annotation_volume()[0]
        self._tree = self._cache.get_structure_tree()
        self._src_to_cre = {
            '23': ['Cux2-IRES-Cre', 'Sepw1-Cre_NP39'],
            '4': ['Nr5a1-Cre'],
            '5it': ['Tlx3-Cre_PL56'],
            '5pt': ['A93-Tg1-Cre', 'Chrna2-Cre_OE25', 'Efr3a-Cre_NO108', 'Sim1-Cre_KJ18'],
            '6': ['Ntsr1-Cre_GN220', 'Syt6-Cre_KI148']
        }
        self._mdl_layers = ['1', '2/3', '4', '5', '6a']
        self._z_midline = 57

    def _region_ids(self, regions, resolve_to_leaf=False):
        if not isinstance(regions, list) or isinstance(regions, numpy.ndarray):
            regions = [regions]
        r_struc = self._tree.get_structures_by_acronym(regions)
        r_ids = numpy.array([_x['id'] for _x in r_struc])
        def resolver(r_ids):
            rslvd = [resolver(_chldr) if len(_chldr) else _base
                     for _base, _chldr in
                     zip(r_ids, self._tree.child_ids(r_ids))]
            return numpy.hstack(rslvd)

        if resolve_to_leaf:
            return resolver(r_ids)
        return r_ids

    def _layer_ids(self, region_acronym, all_layers=False):
        mp = self._tree.get_id_acronym_map()
        lst_layers = self._mdl_layers
        full_acronyms = [region_acronym + _l for _l in lst_layers]
        if all_layers:
            return zip(*[(_l, mp[_a]) if _a in mp
                         else (_l, -1)
                         for _l, _a
                         in zip(lst_layers, full_acronyms)])
        return zip(*[(_l, mp[_a]) for _l, _a
                     in zip(lst_layers, full_acronyms)
                     if _a in mp])

    def classify_profiles(self, profiles, layers):
        from scipy.spatial import distance
        profiles = profiles[numpy.all(~numpy.isnan(profiles), axis=1), :]
        kk = self._profiles.patterns.keys()
        profs = numpy.hstack([self._profiles.patterns[k] for k in kk]).transpose()
        profs = profs[:, numpy.in1d(self._mdl_layers, layers)]
        return numpy.array([kk[numpy.argmin([distance.euclidean(_f, _p)
                                             for _f in profs])]
                            for _p in profiles])

    def layer_profile_for(self, proj_data, tgt_ids, hemi=None):
        if hemi is None:
            v = self._vol
            p = proj_data
        elif hemi == 0:
            v = self._vol[:, :, :self._z_midline]
            p = proj_data[:, :, :self._z_midline]
        else:
            v = self._vol[:, :, self._z_midline:]
            p = proj_data[:, :, self._z_midline:]
        layer_vols = numpy.array([(v == i).sum()
                                  for i in tgt_ids]).astype(float)
        layer_vols = layer_vols / layer_vols.sum()
        layer_sum = numpy.array([p[v == i].sum()
                                 for i in tgt_ids])
        expected = layer_sum.sum() * layer_vols
        return layer_sum / expected

    def tgt_profiles_for_cre(self, src_acronym, cre, all_layers=False, only_ipsi=True):
        tgt_layers, tgt_ids = zip(*[self._layer_ids(_tgt, all_layers=all_layers) for
                                    _tgt in self._mpr.region_names])
        idx_inj = self._region_ids(src_acronym)
        exps = self._cache.get_experiments(injection_structure_ids=idx_inj, cre=cre)
        if only_ipsi:
            hemi = [int(numpy.round(_e['injection_z'] / 100.0) > self._z_midline)
                    for _e in exps]
        else:
            hemi = [None for _ in xrange(len(exps))]
        if len(exps) == 0:
            res_profiles = dict([(_tgt, numpy.empty((0, len(_l))))
                                 for _tgt, _l in zip(self._mpr.region_names, tgt_layers)])
        else:
            proj_data = [self._cache.get_projection_density(_e['id'])[0] for _e in exps]
            res_profiles = dict([(_tgt, numpy.vstack([self.layer_profile_for(_data, _tgt_ids, _hemi)
                                                      for _data, _hemi in zip(proj_data, hemi)]))
                                for _tgt, _tgt_ids in
                                zip(self._mpr.region_names, tgt_ids)])
        res_layers = dict([(_tgt, _layers) for _tgt, _layers in
                           zip(self._mpr.region_names, tgt_layers)])
        return res_profiles, res_layers

    def tgt_profiles_for_src(self, src_acronym, source, **kwargs):
        return self.tgt_profiles_for_cre(src_acronym, self._src_to_cre[source], **kwargs)

    def tgt_profiles(self, source, **kwargs):
        res_profiles = {}
        res_layers = {}
        for m in self._mpr.region_names:
            _res, _layers = self.tgt_profiles_for_src(m, source, **kwargs)
            for k in _res.keys():
                res_profiles[(m, k)] = _res[k]
                res_layers[(m, k)] = _layers[k]
        return res_profiles, res_layers

    def model_profiles_for_src(self, src_acronym, source, layers=None):
        src_id = self._mpr.region2idx(src_acronym)
        res = {}
        for tgt in self._mpr.region_names:
            tgt_id = self._mpr.region2idx(tgt)
            res[tgt] = self._profiles.patterns[self._mxr.max(source, src_id, tgt_id) + 1]
            if layers is not None:
                lyrs = layers[tgt]
                res[tgt] = numpy.hstack([_x for _x, _l in zip(res[tgt], self._mdl_layers)
                                         if _l in lyrs])
        return res

    def model_profiles(self, source, layers=None):
        res = {}
        for src in self._mpr.region_names:
            src_id = self._mpr.region2idx(src)
            for tgt in self._mpr.region_names:
                tgt_id = self._mpr.region2idx(tgt)
                _res = self._profiles.patterns[self._mxr.max(source, src_id, tgt_id) + 1]
                if layers is not None:
                    lyrs = layers[(src, tgt)]
                    _res = numpy.hstack([_x for _x, _l in zip(_res, self._mdl_layers)
                                         if _l in lyrs])
                res[(src, tgt)] = _res
        return res

    @classmethod
    def from_cache(cls, cache):
        from white_matter.wm_recipe.projection_strength import ProjectionStrength
        from white_matter.wm_recipe import layer_profiles
        from white_matter.wm_recipe import region_mapper
        S = ProjectionStrength()
        mxr = layer_profiles.ProfileMixer(S)
        lp = layer_profiles.LayerProfiles()
        mpr = region_mapper.RegionMapper()
        return cls(cache, mpr, mxr, lp)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import mcmodels
    from scipy import stats
    cache = mcmodels.core.VoxelModelCache(manifest_file='/home/reimann/Documents/data/connectivity/voxel_model_manifest.json')
    P = LayerProfileVsRawData.from_cache(cache)
    mpr = P._mpr
    A, layers = zip(*[P.tgt_profiles(_src, all_layers=True, only_ipsi=True) for _src in mpr.source_names])
    B = [P.model_profiles(_src, _layers) for _src, _layers in zip(mpr.source_names, layers)]


    def error_fun(_a, _b):
        epsilon = 0.1
        if _a.shape[0] < 5:
            return numpy.NaN * numpy.ones(_a.shape[1])
        return (_b - numpy.nanmean(_a, axis=0)) / (numpy.nanstd(_a, axis=0) + epsilon)

    E = [dict([(k, error_fun(_A[k], _B[k])) for k in _A.keys()]) for _A, _B in zip(A, B)]

    def plot_error_hist(e, ttl):
        normpdf = lambda _x: stats.norm.pdf(_x) / (numpy.mean(numpy.diff(_x)) * stats.norm.pdf(_x).sum())
        plt.figure()
        bins = numpy.linspace(0, 10, 31)
        err = numpy.vstack([e[(m1,m2)] for m1 in mpr.region_names
                            for m2 in mpr.region_names if m1 != m2])
        H = [numpy.histogram(_err[~numpy.isnan(_err)], bins=bins, density=True)[0] for _err in err.transpose()]
        [plt.plot(bins[:-1], _tmp, label='In layer ' + _lbl) for _tmp, _lbl in zip(H, P._mdl_layers)]
        plt.plot(bins, normpdf(bins), color='black', ls='--', label='Assumed biol. variability')
        plt.legend()
        plt.xlabel('Relative error (in standard deviations)')
        plt.ylabel('Fraction')
        plt.title(ttl)

    for i in range(5):
        for pre in mpr.region_names:
            plt.figure(figsize=(25, 20))
            j = 1
            for post in mpr.region_names:
                if pre == post:
                    continue
                plt.subplot(6, 7, j)
                j += 1
                try:
                    plt.plot(A[i][(pre, post)].transpose(), color='grey', ls='--', lw=0.5)
                    plt.plot(B[i][(pre, post)])
                except:
                    pass
                plt.title(post)
                plt.gca().set_xticks(range(len(P._mdl_layers)))
                plt.gca().set_xticklabels(P._mdl_layers)
            plt.gcf().savefig('overview_%s_%s.pdf' % (mpr.source_names[i], pre))
            plt.close('all')

    for i, src in enumerate(mpr.source_names):
        plot_error_hist(E[i], 'Projections from: %s' % src)
        plt.gcf().savefig('rel_error_%s.pdf' % src)
        plt.close('all')
