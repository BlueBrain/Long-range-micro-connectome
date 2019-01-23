import os


def path_local_to_cfg_root(cfg, lst_args):
    path_local_to_path(cfg, cfg.get("cfg_root", "."), lst_args)


def path_local_to_path(cfg, path, lst_args):
    for _arg in lst_args:
        assert _arg in cfg, "Required argument %s not configured!" % _arg
        if not os.path.isabs(cfg[_arg]):
            cfg[_arg] = os.path.join(path, cfg[_arg])
