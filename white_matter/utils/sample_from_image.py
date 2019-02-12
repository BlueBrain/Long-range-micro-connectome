import scipy.misc
import numpy


# noinspection PyDefaultArgument
class ImgSampler(object):
    def __init__(self, path, fr=None, to=None, cbar=None, cbar_kwargs={}):
        self.img_raw = scipy.misc.imread(path)
        self.cut(fr, to)
        if cbar is not None:
            self.add_cbar(*cbar, **cbar_kwargs)
        else:
            self.cbar_col = None

    def cut(self, fr, to):
        if fr is None:
            fr = [0, 0]
        if to is None:
            to = [self.img_raw.shape[1], self.img_raw.shape[0]]
        self.img = self.img_raw[fr[1]:to[1], fr[0]:to[0], :3]

    def add_cbar(self, x, y, v, filename=None, vals_nan=[]):
        if filename is not None:
            img_cbar = scipy.misc.imread(filename)
        else:
            img_cbar = self.img_raw
        if not hasattr(x, "__iter__"):
            self.cbar_col = img_cbar[y[0]:y[1], x, :3].astype(float)
        else:
            self.cbar_col = img_cbar[y, x[0]:x[1], :3].astype(float)
        dup_vals = numpy.nonzero(numpy.abs(numpy.diff(self.cbar_col, axis=0)).sum(axis=1) == 0)[0]
        u_vals = numpy.setdiff1d(range(len(self.cbar_col)), dup_vals + 1)
        self.cbar_col = self.cbar_col[u_vals]
        if not isinstance(v, dict):
            self.cbar_vals = numpy.linspace(v[0], v[1], self.cbar_col.shape[0])
        else:
            kk = sorted(v.keys())
            self.cbar_vals = numpy.interp(numpy.linspace(0, 1, self.cbar_col.shape[0]),
                                          kk, [v[_k] for _k in kk])
        self.cbar_nans = vals_nan

    def sample(self, nbins_x, nbins_y):
        s_x = numpy.linspace(0, self.img.shape[1], nbins_x + 1)
        s_y = numpy.linspace(0, self.img.shape[0], nbins_y + 1)
        x = (0.5 * (s_x[:-1] + s_x[1:])).astype(int)
        y = (0.5 * (s_y[:-1] + s_y[1:])).astype(int)
        self.sampled = self.img[:, x, :][y]
        if self.cbar_col is not None:
            self.translate()

    def translate(self):
        assert self.cbar_col is not None
        self.translated = numpy.NaN * numpy.ones((self.sampled.shape[0], self.sampled.shape[1]))
        for x in range(self.sampled.shape[1]):
            for y in range(self.sampled.shape[0]):
                v = self.sampled[y, x, :]
                if v.tolist() in self.cbar_nans:
                    self.translated[y, x] = numpy.NaN
                else:
                    d = ((self.cbar_col - v) ** 2).sum(axis=1)
                    mn = numpy.min(d)
                    idxx = numpy.nonzero(d == mn)[0]
                    self.translated[y, x] = numpy.mean(self.cbar_vals[idxx])
        self.out = self.translated

    def scale_to_target(self, tgt_x, tgt_y, tgt_val):
        scalar = tgt_val / self.translated[tgt_y, tgt_x]
        self.scaled = scalar * self.translated
        self.out = self.scaled

    def map(self, func):
        self.out = func(self.out)

    def condense(self, idx_fr, idx_to, func=numpy.nansum):
        z_fr = zip(idx_fr[:-1], idx_fr[1:])
        z_to = zip(idx_to[:-1], idx_to[1:])
        return numpy.array([[func(self.out[fr[0]:fr[1], t[0]:t[1]])
                             for t in z_to] for fr in z_fr])
