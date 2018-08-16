def read_config(fn):
    import json
    with open(fn, 'r') as fid:
        ret = json.load(fid)
    return ret["SynapseTypes"]


class SynapseTypes(object):
    def __init__(self, cfg_file=None):
        if cfg_file is None:
            import os
            cfg_file = os.path.join(os.path.split(__file__)[0], 'default.json')
        self.cfg = read_config(cfg_file)
        self.cfg["synapse_type_yaml"] = self._treat_path(self.cfg["synapse_type_yaml"])

    @staticmethod
    def _treat_path(fn):
        import os
        if not os.path.isabs(fn):
            fn = os.path.join(os.path.split(__file__)[0], fn)
        return fn

    def __getitem__(self, item):
        return self.cfg["synapse_type_mapping"][item]

