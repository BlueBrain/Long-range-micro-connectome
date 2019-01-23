def read_config(fn):
    import json
    with open(fn, 'r') as fid:
        ret = json.load(fid)
    return ret["SynapseTypes"]


class SynapseTypes(object):
    def __init__(self, cfg_file=None):
        import os
        from white_matter.utils.paths_in_config import path_local_to_path
        if cfg_file is None:
            cfg_file = os.path.join(os.path.split(__file__)[0], 'default.json')
        self.cfg = read_config(cfg_file)
        local_path = os.path.split(__file__)[0]
        path_local_to_path(self.cfg, local_path, ["synapse_type_yaml"])

    def __getitem__(self, item):
        return self.cfg["synapse_type_mapping"][item]

