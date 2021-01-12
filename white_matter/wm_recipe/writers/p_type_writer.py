import numpy


class PTypeWriter(object):
    def __init__(self, mpr, namer, p_type_mdls, proj_str, interaction_thresh=2.5):
        self.p_type_mdls = p_type_mdls
        self.mpr = mpr
        self.namer = namer
        self.proj_str = proj_str
        self.thresh = interaction_thresh

    @staticmethod
    def _nums2str(v, prefix='\t\t'):
        def line(v):
            if len(v) == 0:
                return ''
            ret = "%5.3f" % v[0]
            for _v in v[1:]:
                ret = ret + ', ' + ("%5.3f" % _v)
            return ret
        n_chunks = int(numpy.maximum(len(v) / 10, 1))
        chunks = numpy.split(v, numpy.linspace(0, len(v), n_chunks + 1)[1:-1].astype(int))
        ret = line(chunks[0])
        for chunk in chunks[1:]:
            ret = ret + ',\n' + prefix + line(chunk)
        return ret

    def __call__(self, fid):
        def single_entry(population, lst_tgts, lst_fractions, interact_tgts, interact_vals):
            assert len(interact_vals) == (len(interact_tgts) * (len(interact_tgts) - 1) / 2)
            fid.write('\t- population: %s\n' % population)
            fid.write('\t  fractions:\n')
            for tgt, frac in zip(lst_tgts, lst_fractions):
                fid.write('\t\t  %s: %f\n' % (tgt, frac))
            if len(interact_vals) > 0:
                fid.write('\t  interaction_mat:\n')
                fid.write('\t\t  projections: [%s' % interact_tgts[0])
                for tgt in interact_tgts[1:]:
                    fid.write(',\n\t\t                %s' % tgt)
                fid.write(']\n')
                fid.write('\t\t  strengths: [' + self._nums2str(interact_vals,
                                                                prefix='\t\t              ')
                          + ']\n')
            fid.write('\n')

        fid.write("p-types:\n")
        for source_name in self.mpr.source_names:
            p_type_mdl = self.p_type_mdls[source_name]
            src_valid = numpy.hstack([self.proj_str(src_type=source_name, hemi=hemi,
                                                    measurement='connection density') > 0
                                      for hemi in ['ipsi', 'contra']])
            src_mat = p_type_mdl.first_order_mat()
            for reg_from in self.mpr.region_names:
                print("Getting interactions for %s" % reg_from)
                M_i = p_type_mdl.interaction_mat(reg_from, no_redundant=True)
                M_i[M_i < self.thresh] = 1.0

                proj_list = [self.namer.projection(reg_from, source_name, _x[0], hemi=_x[1])
                             for _x in p_type_mdl.region_hemi_names()]
                proj_fracs = src_mat[self.mpr.region2idx(reg_from)]
                assert not numpy.any(
                    src_valid[self.mpr.region2idx(reg_from)] &
                    (proj_fracs <= 0)
                )
                valid1 = (proj_fracs > 0)
                if not numpy.any(valid1):
                    print("Nothing for %s, %s" % (reg_from, source_name))
                    continue
                subM = M_i[:, valid1][valid1]
                proj_list = [_x for _v, _x in zip(valid1, proj_list) if _v]
                proj_fracs = proj_fracs[valid1]
                valid2 = numpy.any(subM >= self.thresh, axis=0) | numpy.any(subM >= self.thresh, axis=1)
                interact_list = [_x for _v, _x in zip(valid2, proj_list) if _v]
                interact_vals = subM[:, valid2][valid2]
                interact_vals = interact_vals[numpy.triu_indices_from(interact_vals, 1)]
                single_entry(self.namer.comb_pop(reg_from, source_name), proj_list,
                             proj_fracs, interact_list, interact_vals)
            fid.write("\n")
        fid.write("\n")
