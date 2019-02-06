#!/usr/bin/env python
import numpy
from white_matter.wm_recipe.parcellation import RegionMapper
from white_matter.utils.data_from_config import read_config as read_config_default


def read_cfg(cfg_file):
    from white_matter.utils.paths_in_config import path_local_to_path

    cfg = read_config_default(cfg_file)
    cfg_root = cfg["cfg_root"]
    for k in cfg["PTypes"].keys():
        path_local_to_path(cfg["PTypes"][k], cfg_root, ["json_cache", "h5_cache"])
    return cfg


def src_mat(proj_str, src, measurement):
    predict_fun = lambda x: numpy.minimum(numpy.sqrt(x) / 4, 1.0)  # TODO: read from config
    M = numpy.vstack([numpy.hstack([proj_str(src_type=src, hemi=hemi,
                                    measurement=measurement)
                                    for hemi in lst_hemi])
                     for lst_hemi in [['ipsi', 'contra'], ['contra', 'ipsi']]])
    return predict_fun(M)


def make_model_for_source(S, src, cfg, **kwargs):
    from white_matter.wm_recipe.p_types.ptype_tree_model import TreeInnervationModel
    F_topo = src_mat(S, src, cfg["mat_tree_topology"])
    F = src_mat(S, src, cfg["mat_predict_innervation"])
    mdl = TreeInnervationModel.from_con_mats(F_topo, F, **kwargs)
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
    mpr = RegionMapper(cfg["BrainParcellation"])
    cfg = cfg["PTypes"]
    for src in cfg.keys():
        T, M = make_model_for_source(S, src, cfg[src], mpr=mpr)
        write_model_for_source(T, M, cfg[src])


if __name__ == "__main__":
    import sys
    import json
    import os
    from white_matter.wm_recipe.projection_strength import ProjectionStrength
    cfg = read_cfg(sys.argv[1])
    if len(sys.argv) > 2:
        S = ProjectionStrength(cfg_file=sys.argv[2])
    else:
        S = ProjectionStrength(cfg_file=sys.argv[1])
    main(S, cfg)


