from copy_writer import CopyWriter


class SynapseTypeWriter(CopyWriter):
    def __init__(self, syn_types):
        fn = syn_types.cfg["synapse_type_yaml"]
        super(SynapseTypeWriter, self).__init__(fn)
