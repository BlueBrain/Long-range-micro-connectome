from ...utils.data_from_config import read_config


class SynapseTypes(object):
    def __init__(self, cfg_file=None):
        import os
        from white_matter.utils.paths_in_config import path_local_to_cfg_root
        if cfg_file is None:
            cfg_file = os.path.join(os.path.split(__file__)[0], 'default.json')
        cfg = read_config(cfg_file)
        self.cfg = cfg["SynapseTypes"]
        self.cfg["cfg_root"] = cfg["cfg_root"]
        path_local_to_cfg_root(self.cfg, ["synapse_type_yaml"])

    def __getitem__(self, item):
        return self.cfg["synapse_type_mapping"][item]

