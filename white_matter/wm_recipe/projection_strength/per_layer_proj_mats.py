import numpy
from white_matter.wm_recipe.region_mapper import RegionMapper
from white_matter.wm_recipe.sample_from_image import ImgSampler


def deactivate_where_volume_is_zero(mats, vol_dict, mpr):
    for k, mat in mats.items():
        d = dict(k)
        for i, reg in enumerate(mpr.region_names):
            src_str = d['src_type']
            if src_str.startswith('5'):
                src_str = '5'
            if vol_dict[reg][src_str] == 0:
                mat[i, :] = 0.0



def per_layer_proj_mats(cfg, M_i, M_c, scale=True, vol_dict=None):
    mpr = RegionMapper()
    per_layer_cbar = (tuple(cfg["cbar_width"]), cfg["cbar_height"], tuple(cfg["cbar_values"]))
    per_layer_cbar_kwargs = cfg["cbar_kwargs"]
    per_layer_cbar_kwargs["filename"] = cfg["cbar_filename"]
    per_layer_mdl_idx_fr = cfg["per_layer_module_separators"]
    per_layer_N = cfg["per_layer_mat_height"]
    per_layer_mdl_idx_to = numpy.unique(mpr.module_idx.values())
    N = mpr.n_regions()
    frac_lost_in_thresh = cfg["threshold_fraction"]
    fn_pat_ipsi = str(cfg["per_layer_filename_ipsi"])
    fn_pat_contra = str(cfg["per_layer_filename_contra"])

    def dictmap(d, func):
        return dict([(k, func(v)) for k, v in d.items()])

    '''READ THE MAIN CONNECTION MATRIX'''
    '''import h5py, os
    h5 = h5py.File(os.path.join(os.path.split(__file__)[0], 'digested/connection_matrices.h5'), 'r')
    M_i = numpy.array(h5['wild_type/ipsi/connection density'])
    M_c = numpy.array(h5['wild_type/contra/connection density'])'''
    '''SCALE THE MATRIX SUCH THAT THE VALUE FOR SSp-ll to SSp-ll MATCHES THE SPECIFIED VALUE'''
    if scale:
        scalar = cfg["scaling"]["value"] / M_i[mpr.region2idx(str(cfg["scaling"]["region"])),
                                               mpr.region2idx(str(cfg["scaling"]["region"]))]
        M_i = scalar * M_i
        M_c = scalar * M_c

    '''READ THE LAYER-SPECIFIC MATRICES'''
    fn_ipsi = dict([(k, fn_pat_ipsi % k) for k in mpr.source_names])
    fn_ipsi['master'] = fn_pat_ipsi % 'ALL'
    fn_contra = dict([(k, fn_pat_contra % k) for k in mpr.source_names])
    fn_contra['master'] = fn_pat_contra % 'ALL'
    m_ipsi = dictmap(fn_ipsi, lambda fn: ImgSampler(fn, cbar=per_layer_cbar, cbar_kwargs=per_layer_cbar_kwargs))
    m_contra = dictmap(fn_contra, lambda fn: ImgSampler(fn, cbar=per_layer_cbar, cbar_kwargs=per_layer_cbar_kwargs))
    '''LAYER SPECIFIC MATRICES ARE ONLY SAMPLED AT 43 BY 27'''
    [_x.sample(N, per_layer_N) for _x in m_ipsi.values()]
    [_x.sample(N, per_layer_N) for _x in m_contra.values()]
    '''LAYER SPECIFIC MATRICES LACK THE PLUS 1 IN THE LOG TRANSFORM, INSTEAD HAVE ZEROS MASKED OUT'''
    [_x.map(lambda x: 10**x) for _x in m_ipsi.values()]
    [_x.map(lambda x: 10**x) for _x in m_contra.values()]
    '''CONDENSE LAYER SPECIFIC MATRICES TO MODULE PATHWAYS (6 BY 6)'''
    ss_ipsi = dictmap(m_ipsi, lambda x: x.condense(per_layer_mdl_idx_fr, per_layer_mdl_idx_to, func=numpy.nansum))
    ss_contra = dictmap(m_contra, lambda x: x.condense(per_layer_mdl_idx_fr, per_layer_mdl_idx_to, func=numpy.nansum))

    '''NORMALIZE BY THE SUM OF ALL LAYER SPECIFIC MATRICES. ASSUMPTION: TOTAL STRENGTH IS SUM OF PATHWAYS FROM INDIVIDUAL LAYERS'''
    nrmlz_ipsi = numpy.dstack([ss_ipsi[k] for k in mpr.source_names]).sum(axis=2)
    nrmlz_contra = numpy.dstack([ss_contra[k] for k in mpr.source_names]).sum(axis=2)
    I = dictmap(ss_ipsi, lambda x: x / nrmlz_ipsi)
    C = dictmap(ss_contra, lambda x: x / nrmlz_contra)

    def scaled_submats(M, scales):
        ret = []
        for mdl_s, row in zip(mpr.module_names, scales):
            out_row = []
            for mdl_t, v in zip(mpr.module_names, row):
                out_row.append( v * M[:, mpr.module2idx(mdl_t)][mpr.module2idx(mdl_s)])
            ret.append(numpy.hstack(out_row))
        return numpy.vstack(ret)
    '''FINAL RESULT IS THEN TAKING THE MODULE SPECIFIC SUBMATRICES FROM THE MAIN MATRIX, SCALED BY FRACTIONS CALCULATED FOR EACH MODULE PATHWAY'''

    FINAL = dict([((('hemi', 'ipsi'), ('src_type', k)), scaled_submats(M_i, I[k])) for k in mpr.source_names]
                 + [((('hemi', 'contra'), ('src_type', k)), scaled_submats(M_c, C[k])) for k in mpr.source_names])
    if vol_dict is not None:
        deactivate_where_volume_is_zero(FINAL, vol_dict, mpr)

    '''APPLY CUTOFF TO REMOVE VERY WEAK PROJECTIONS. CUTOFF SELECTED SUCH THAT 5% OF THE TOTAL DENSITY IS LOST'''
    all_str_vals = numpy.hstack([x.flatten() for x in FINAL.values()])
    all_str_vals.sort()
    cutoff = all_str_vals[numpy.nonzero((numpy.cumsum(all_str_vals) / numpy.nansum(all_str_vals)) > frac_lost_in_thresh)[0][0]]

    def apply_cutoff(X):
        X[X < cutoff] = 0.0
        X[numpy.isnan(X)] = 0.0
    dictmap(FINAL, apply_cutoff)
    return FINAL

