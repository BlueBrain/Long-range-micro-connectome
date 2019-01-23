import mcmodels
import numpy
from white_matter.wm_recipe import region_mapper
M = region_mapper.RegionMapper()


def make_voxel_model(cfg):
    cache = mcmodels.core.VoxelModelCache(manifest_file=cfg["cache_manifest"])
    voxel_array, source_mask, target_mask = cache.get_voxel_connectivity_array()
    return voxel_array, source_mask, target_mask,\
           cache.get_structure_tree(), cache.get_annotation_volume()[0]


def get_layer_specific_ids(tree, vol):
    r_id = [_r['id'] for _r in tree.get_structures_by_acronym(M.region_names)]
    descendants = numpy.hstack(tree.child_ids(r_id))
    return descendants[numpy.in1d(descendants, vol)]


def make_regionalized_model(voxel_array, source_mask, target_mask, tree, vol, hemi,
                            per_layer=True):
    from mcmodels.models.voxel import RegionalizedModel
    r_ids = [_r['id'] for _r in tree.get_structures_by_acronym(M.region_names)]
    if per_layer:
        s_key = source_mask.get_key(get_layer_specific_ids(tree, vol),
                                    hemisphere_id=2)
    else:
        s_key = source_mask.get_key(r_ids, hemisphere_id=2)
    if hemi == 'ipsi':
        t_key = target_mask.get_key(r_ids, hemisphere_id=2)
    else:
        t_key = target_mask.get_key(r_ids, hemisphere_id=1)
    return RegionalizedModel.from_voxel_array(voxel_array, s_key, t_key)


def layer_specific_matrix(mdl, tree, connection_type, layers):
    '''We ended up not using this. Injections are too course-grained to yield usable results.
    Still, this is how it would have been done'''
    A = mdl.__getattribute__(connection_type)
    sources = tree.get_structures_by_id(mdl.source_regions)
    source_names = [_s['acronym'] for _s in sources]
    targets = tree.get_structures_by_id(mdl.target_regions)
    target_names = [_t['acronym'] for _t in targets]
    target_idx = numpy.array([target_names.index(_r) for _r in M.region_names])
    res = []
    w = []
    for l in layers:
        tmp_res = numpy.zeros((len(M.region_names), len(M.region_names)))
        tmp_w = numpy.zeros((len(M.region_names)))
        compound_src = [_r + l for _r in M.region_names]
        tmp_idx, source_idx = zip(*[(i, source_names.index(_r))
                                    for i, _r in enumerate(compound_src)
                                    if _r in source_names])
        assert len(tmp_idx) > 0, "Invalid layer specification: %s" % l
        tmp_res[numpy.array(tmp_idx), :] = A[:, target_idx][numpy.array(source_idx)]
        tmp_w[numpy.array(tmp_idx)] = mdl.source_counts[numpy.array(source_idx)]
        res.append(tmp_res)
        w.append(tmp_w)
    if 'normalized' in connection_type:
        w = (numpy.vstack(w).astype(float) / numpy.vstack(w).sum(axis=0)).transpose()
        return numpy.dstack([w[:, i:(i+1)] * R for i, R in enumerate(res)]).sum(axis=2)
    else:
        return numpy.dstack(res).sum(axis=2)


def region_specific_matrix(mdl, tree, connection_type):
    A = mdl.__getattribute__(connection_type)
    sources = tree.get_structures_by_id(mdl.source_regions)
    source_names = [_s['acronym'] for _s in sources]
    source_idx = numpy.array([source_names.index(_r) for _r in M.region_names])
    targets = tree.get_structures_by_id(mdl.target_regions)
    target_names = [_t['acronym'] for _t in targets]
    target_idx = numpy.array([target_names.index(_r) for _r in M.region_names])
    return A[:, target_idx][source_idx]


def scale_to_target(cfg, Mtrx):
    if "scaling" in cfg:
        scale_tgt_name = cfg["scaling"].get("measurement", "connection density")
        scale_hemi = cfg["scaling"].get("hemisphere", "ipsi")
        tgt_key = (('src_type', 'wild_type'), ('hemi', scale_hemi), ('measurement', scale_tgt_name))
        ipsi_key = (('src_type', 'wild_type'), ('hemi', 'ipsi'), ('measurement', scale_tgt_name))
        contra_key = (('src_type', 'wild_type'), ('hemi', 'contra'), ('measurement', scale_tgt_name))
        idx = M.region2idx(cfg["scaling"]["region"])
        factor = cfg["scaling"]["value"] / Mtrx[tgt_key][idx, idx]
        Mtrx[ipsi_key] = factor * Mtrx[ipsi_key]
        Mtrx[contra_key] = factor * Mtrx[contra_key]


def master_proj_mats(cfg):
    voxel_array, source_mask, target_mask, tree, vol = make_voxel_model(cfg)
    mdl_i = make_regionalized_model(voxel_array, source_mask, target_mask, tree, vol,
                                    hemi='ispi', per_layer=False)
    mdl_c = make_regionalized_model(voxel_array, source_mask, target_mask, tree, vol,
                                    hemi='contra', per_layer=False)
    measurements = ['connection strength', 'connection density',
                    'normalized connection strength',
                    'normalized connection density']
    Mtrx = {}
    src_type = (('src_type', 'wild_type'),)

    for msrmnt in measurements:
        Mtrx[src_type + (('hemi', 'ipsi'), ('measurement', msrmnt))] = region_specific_matrix(mdl_i, tree, msrmnt)
        Mtrx[src_type + (('hemi', 'contra'), ('measurement', msrmnt))] = region_specific_matrix(mdl_c, tree, msrmnt)
    scale_to_target(cfg, Mtrx)
    return Mtrx
