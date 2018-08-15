from copy_writer import CopyWriter


class SynapseTypeWriter(CopyWriter):
    def __init__(self):
        import os
        fn = os.path.join(os.path.split(__file__)[0],
                          '../yaml/synapse_type.yaml')
        super(SynapseTypeWriter, self).__init__(fn)
