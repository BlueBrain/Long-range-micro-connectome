def main(cfg):
    import h5py, os
    from white_matter.wm_recipe.projection_strength.master_proj_mats import master_proj_mats
    from white_matter.wm_recipe.projection_strength import ProjectionStrength
    cfg_root = cfg["cfg_root"]
    out_h5_fn = cfg["h5_cache"]
    if not os.path.isabs(out_h5_fn):
        out_h5_fn = os.path.join(cfg_root, out_h5_fn)
    if not os.path.exists(os.path.split(out_h5_fn)[0]):
        os.makedirs(os.path.split(out_h5_fn)[0])
    if os.path.exists(out_h5_fn):
        h5 = h5py.File(out_h5_fn, 'r+')
    else:
        h5 = h5py.File(out_h5_fn, 'w')
    try:
        M = master_proj_mats(cfg)
        for k, v in M.items():
            h5.require_dataset(ProjectionStrength._dict_to_path(dict(k)), v.shape,
                               float, data=v)
    except:
        raise
    finally:
        h5.close()


if __name__ == "__main__":
    import sys
    import json
    import os
    cfg_file = sys.argv[1]
    with open(cfg_file, 'r') as fid:
        cfg = json.load(fid)["ProjectionStrength"]
    cfg["cfg_root"] = os.path.split(cfg_file)[0]
    main(cfg)
