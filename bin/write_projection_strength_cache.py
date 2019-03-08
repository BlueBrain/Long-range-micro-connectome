#!/usr/bin/env python


def main(cfg):
    import h5py, os
    from white_matter.wm_recipe.projection_strength.master_proj_mats import master_proj_mats
    from white_matter.wm_recipe.projection_strength import ProjectionStrength
    from white_matter.wm_recipe.parcellation import RegionMapper
    mpr = RegionMapper(cfg["BrainParcellation"])
    relevant_chapter = cfg["ProjectionStrength"]
    out_h5_fn = relevant_chapter["h5_cache"]
    if not os.path.exists(os.path.split(out_h5_fn)[0]):
        os.makedirs(os.path.split(out_h5_fn)[0])
    if os.path.exists(out_h5_fn):
        h5 = h5py.File(out_h5_fn, 'r+')
    else:
        h5 = h5py.File(out_h5_fn, 'w')
    try:
        M = master_proj_mats(relevant_chapter, mpr)
        for k, v in M.items():
            h5.require_dataset(ProjectionStrength._dict_to_path(dict(k)), v.shape,
                               float, data=v)
    except:
        raise
    finally:
        h5.close()


if __name__ == "__main__":
    import sys
    import os
    from white_matter.utils.paths_in_config import path_local_to_cfg_root
    from white_matter.utils.data_from_config import read_config
    cfg_file = sys.argv[1]
    cfg = read_config(cfg_file)
    cfg["ProjectionStrength"]["cfg_root"] = cfg["cfg_root"]
    path_local_to_cfg_root(cfg["ProjectionStrength"],
                           ["cache_manifest", "h5_cache"])
    main(cfg)
