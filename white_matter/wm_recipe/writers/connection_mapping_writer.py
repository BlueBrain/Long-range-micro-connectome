from copy_writer import CopyWriter


class ConnectionMappingWriter(CopyWriter):
    def __init__(self):
        import os
        fn = os.path.join(os.path.split(__file__)[0],
                          '../yaml/mapping.yaml')
        super(ConnectionMappingWriter, self).__init__(fn)
