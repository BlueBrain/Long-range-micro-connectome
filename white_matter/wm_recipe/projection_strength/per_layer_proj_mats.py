import numpy
from white_matter.utils.data_from_config import ConfiguredDataSource
'''The strategy to combine a wild type connection matrix with digitized 
data for projection-class-specific projection strength to derive dense
projection-class-specific matrices'''


def deactivate_where_volume_is_zero(mats, vol_dict, mpr):
    for k, mat in mats.items():
        d = dict(k)
        for i, reg in enumerate(mpr.region_names):
            src_str = d['src_type']
            if src_str.startswith('5'):
                src_str = '5'
            if vol_dict[reg][src_str] == 0:
                mat[i, :] = 0.0


class ProjectionClassSpecificMat(ConfiguredDataSource):
    relevant_chapter = "ProjectionStrength"
    relevant_section = "per_projection_class_ipsi"

    def __init__(self, cfg, mpr):
        self.source_names = mpr.source_names
        super(ProjectionClassSpecificMat, self).__init__(cfg)
        self.parameterize(self.cfg)
        self.patterns = dict([(k, 10 ** v) for k, v in self.patterns.items()])

    def __pattern_to_filenames__(self, pat):
        ret = dict([(k, pat % k) for k in self.source_names])
        ret['master'] = pat % 'ALL'
        return ret


class ProjectionClassSpecificMatC(ProjectionClassSpecificMat):
    relevant_section = 'per_projection_class_contra'


def per_layer_proj_mats(cfg, mpr, M_i, M_c, scale=True, vol_dict=None):
    per_layer_mdl_idx_fr = cfg["module_separators_source"]
    per_layer_mdl_idx_to = cfg["module_separators_target"]
    frac_lost_in_thresh = cfg["threshold_fraction"]

    def dictmap(d, func):
        return dict([(k, func(v)) for k, v in d.items()])

    '''SCALE THE MATRIX SUCH THAT THE VALUE FOR THE SPECIFIED REGION MATCHES THE SPECIFIED VALUE'''
    if scale:
        scalar = cfg["scaling"]["value"] / M_i[mpr.region2idx(str(cfg["scaling"]["region"])),
                                               mpr.region2idx(str(cfg["scaling"]["region"]))]
        M_i = scalar * M_i
        M_c = scalar * M_c

    '''GENERATE LAYER-SPECIFIC MATRICES'''
    m_ipsi = ProjectionClassSpecificMat(cfg, mpr)
    m_contra = ProjectionClassSpecificMatC(cfg, mpr)
    '''CONDENSE LAYER SPECIFIC MATRICES TO MODULE PATHWAYS (6 BY 6)'''
    m_ipsi.condense(per_layer_mdl_idx_fr, per_layer_mdl_idx_to, func=numpy.nansum)
    m_contra.condense(per_layer_mdl_idx_fr, per_layer_mdl_idx_to, func=numpy.nansum)
    ss_ipsi = m_ipsi.patterns
    ss_contra = m_contra.patterns
    '''NORMALIZE BY THE SUM OF ALL LAYER SPECIFIC MATRICES. ASSUMPTION:
    TOTAL STRENGTH IS SUM OF PATHWAYS FROM INDIVIDUAL LAYERS'''
    nrmlz_ipsi = numpy.dstack([ss_ipsi[k] for k in mpr.source_names]).sum(axis=2)
    nrmlz_contra = numpy.dstack([ss_contra[k] for k in mpr.source_names]).sum(axis=2)
    I = dictmap(ss_ipsi, lambda x: x / nrmlz_ipsi)
    C = dictmap(ss_contra, lambda x: x / nrmlz_contra)

    def scaled_submats(M, scales):
        ret = []
        for mdl_s, row in zip(mpr.module_names, scales):
            out_row = []
            for mdl_t, v in zip(mpr.module_names, row):
                out_row.append(v * M[:, mpr.module2idx(mdl_t)][mpr.module2idx(mdl_s)])
            ret.append(numpy.hstack(out_row))
        return numpy.vstack(ret)
    '''FINAL RESULT IS THEN TAKING THE MODULE SPECIFIC SUBMATRICES FROM THE MAIN MATRIX,
    SCALED BY FRACTIONS CALCULATED FOR EACH MODULE PATHWAY'''

    FINAL = dict([((('hemi', 'ipsi'), ('src_type', k)), scaled_submats(M_i, I[k])) for k in mpr.source_names]
                 + [((('hemi', 'contra'), ('src_type', k)), scaled_submats(M_c, C[k])) for k in mpr.source_names])
    if vol_dict is not None:
        deactivate_where_volume_is_zero(FINAL, vol_dict, mpr)

    '''APPLY CUTOFF TO REMOVE VERY WEAK PROJECTIONS. CUTOFF SELECTED SUCH THAT 5% OF THE TOTAL DENSITY IS LOST'''
    def set_cutoff(lst_keys):
        all_str_vals = numpy.hstack([FINAL[k].flatten() for k in lst_keys])
        all_str_vals.sort()
        cutoff = all_str_vals[numpy.nonzero((numpy.cumsum(all_str_vals) /
                                             numpy.nansum(all_str_vals)) > frac_lost_in_thresh)[0][0]]
        for k in lst_keys:
            FINAL[k][FINAL[k] < cutoff] = 0.0
            FINAL[k][numpy.isnan(FINAL[k])] = 0.0
    '''Separately for cortex and thalamus'''
    set_cutoff([k for k in FINAL.keys() if k[1][1] == 'tc'])
    set_cutoff([k for k in FINAL.keys() if k[1][1] != 'tc'])
    return FINAL

