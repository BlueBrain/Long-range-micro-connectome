class CopyWriter(object):
    def __init__(self, fn_to_copy):
        self.fn = fn_to_copy

    def __call__(self, fid):
        with open(self.fn, 'r') as fid_in:
            for ln in fid_in.readlines():
                fid.write(ln)
