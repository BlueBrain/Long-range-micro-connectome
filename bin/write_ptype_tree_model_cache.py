#!/usr/bin/env python
import numpy
from white_matter.wm_recipe import region_mapper

mpr = region_mapper.RegionMapper()


def load_cfg(cfg_file):
    import json, os

    def __treat_path(fn):
        if not os.path.isabs(fn):
            fn = os.path.join(os.path.split(cfg_file)[0], fn)
        return fn

    with open(cfg_file, 'r') as fid:
        cfg = json.load(fid)["PTypes"]
    for k in cfg.keys():
        cfg[k]["json_cache"] = __treat_path(cfg[k]["json_cache"])
        cfg[k]["h5_cache"] = __treat_path(cfg[k]["h5_cache"])
    return cfg


def src_mat(proj_str, src, measurement):
    predict_fun = lambda x: numpy.minimum(numpy.sqrt(x) / 4, 1.0)  # TODO: read from config
    M = numpy.vstack([numpy.hstack([proj_str(src_type=src, hemi=hemi,
                                    measurement=measurement)
                                    for hemi in lst_hemi])
                     for lst_hemi in [['ipsi', 'contra'], ['contra', 'ipsi']]])
    return predict_fun(M)


def make_model_for_source(S, src, cfg):
    from white_matter.wm_recipe.p_types.ptype_tree_model import TreeInnervationModel
    F_topo = src_mat(S, src, cfg["mat_tree_topology"])
    F = src_mat(S, src, cfg["mat_predict_innervation"])
    mdl = TreeInnervationModel.from_con_mats(F_topo, F)
    return mdl.T, mdl._val_mask


def write_model_for_source(T, F, cfg):
    import json, os, h5py
    import networkx as nx
    fn_json = cfg["json_cache"]
    fn_h5 = cfg["h5_cache"]
    dset_h5 = str(cfg["h5_dset"])
    with open(fn_json, 'w') as fid:
        json.dump(nx.node_link_data(T), fid)
    if os.path.exists(fn_h5):
        h5 = h5py.File(fn_h5, 'a')
    else:
        h5 = h5py.File(fn_h5, 'w')
    h5.require_dataset(dset_h5, F.shape, F.dtype)
    h5[dset_h5][:] = F
    h5.close()


def main(S, cfg):
    for src in cfg.keys():
        T, M = make_model_for_source(S, src, cfg[src])
        write_model_for_source(T, M, cfg[src])


if __name__ == "__main__":
    import sys
    import json
    import os
    from white_matter.wm_recipe.projection_strength import ProjectionStrength
    cfg_ptypes = load_cfg(sys.argv[1])
    if len(sys.argv) > 2:
        S = ProjectionStrength(cfg_file=sys.argv[2])
    else:
        S = ProjectionStrength()
    main(S, cfg_ptypes)


