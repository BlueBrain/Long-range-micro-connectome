import numpy
import logging

logging.basicConfig(level=1)
info_log = logging.getLogger(__file__)


class BarycentricCoordinates(object):

    def __init__(self, x, y):
        assert len(x) == 3 and len(y) == 3
        self._x = x
        self._y = y
        self._S = numpy.matrix([x, y]).transpose()
        self._T = numpy.matrix([[self._x[0] - self._x[2], self._x[1] - self._x[2]],
                                [self._y[0] - self._y[2], self._y[1] - self._y[2]]])

    def cart2bary(self, x, y):
        lx = x - self._x[2]
        ly = y - self._y[2]
        res = numpy.linalg.solve(self._T, numpy.vstack([lx, ly]))
        return numpy.vstack([res[0, :], res[1, :], 1.0 - res.sum(axis=0)]).transpose()

    def bary2cart(self, a, b, c):
        return numpy.array(numpy.vstack([a, b, c]).transpose() * self._S)

    def area(self):
        p1 = numpy.array([self._x[0] - self._x[2], self._y[0] - self._y[2]])
        p2 = numpy.array([self._x[1] - self._x[2], self._y[1] - self._y[2]])
        l1 = numpy.linalg.norm(p1)
        l2 = numpy.linalg.norm(p2)
        assert l1 > 0 and l2 > 0
        return numpy.sin(numpy.arccos(numpy.sum(p1 * p2) / (l1 * l2))) * (l1 * l2) * 0.5


class BarycentricColors(BarycentricCoordinates):

    # noinspection PyDefaultArgument
    def __init__(self, x, y, red=[1, 0, 0], green=[0, 1, 0], blue=[0, 0, 1]):
        super(BarycentricColors, self).__init__(x, y)
        self._cols = numpy.matrix(numpy.vstack([red, green, blue]).transpose())

    def col(self, x, y):
        b = self.cart2bary(x, y)
        b[b > 1.0] = 1.0
        b[b < 0.0] = 0.0
        return numpy.array((self._cols * b.transpose()).transpose())

    def img(self, mask, convolve_var=None):
        nz = numpy.nonzero(mask)
        out_img = numpy.zeros(mask.shape + (3,))
        out_img[nz[0], nz[1], :] = self.col(nz[1].astype(float), nz[0].astype(float))
        if convolve_var is not None:
            from scipy.stats import norm
            from scipy.signal import convolve2d
            sd = numpy.minimum(numpy.sqrt(convolve_var), 100)
            if sd < numpy.sqrt(convolve_var):
                info_log.info("\tExcessive mapping variance found! Reducing to 100!")
            info_log.info("\t\tConvolving final mapping with: %f" % sd)
            X, Y = numpy.meshgrid(numpy.arange(-2 * sd, 2 * sd), numpy.arange(-2 * sd, 2 * sd))
            kernel = norm(0, sd).pdf(numpy.sqrt(X ** 2 + Y ** 2))
            c_mask = convolve2d(mask, kernel, 'same')
            for i in range(3):
                out_img[:, :, i] = convolve2d(out_img[:, :, i], kernel, 'same')\
                                   / c_mask
            out_img[~mask, :] = 0
        return out_img

    # noinspection PyTypeChecker
    def show_img(self, mask, zoom=True, sz_x=8, show_poles=True, convolve_var=None):
        nz = numpy.nonzero(mask)
        out_img = self.img(mask, convolve_var=convolve_var)
        if zoom:
            y1, y2 = nz[0].min(), nz[0].max() + 1
            x1, x2 = nz[1].min(), nz[1].max() + 1
            out_img = out_img[y1:y2, x1:x2, :]
        else:
            x1 = 0; x2 = out_img.shape[1]
            y1 = 0; y2 = out_img.shape[0]
        from matplotlib import pyplot as plt
        sz_y = sz_x * (y2 - y1) / (x2 - x1)
        fig = plt.figure(figsize=(sz_x, sz_y))
        ax = fig.add_axes([0, 0, 1, 1])
        plt.axis('off')
        ax.imshow(out_img, extent=(x1-0.5, x2-0.5, y2-0.5, y1-0.5))
        ax.contour(mask[y1:y2, x1:x2], 1,
                   extent=(x1-0.5, x2-0.5, y1-0.5, y2-0.5),
                   colors=['w'])
        if show_poles:
            for i in range(3):
                ax.plot(self._x[i], self._y[i], 'v',
                        color=[1.0, 1.0, 1.0], markersize=15)
                ax.plot(self._x[i], self._y[i], 'v',
                        color=numpy.array(self._cols)[:, i],
                        markersize=10)

        if convolve_var is not None:
            cx = numpy.mean(ax.get_xlim())
            cy = numpy.mean(ax.get_ylim())
            xx = cx + numpy.sqrt(convolve_var) * numpy.cos(numpy.linspace(0, 2 * numpy.pi, 100))
            yy = cy + numpy.sqrt(convolve_var) * numpy.sin(numpy.linspace(0, 2 * numpy.pi, 100))
            ax.plot(xx, yy, color='grey', ls='--')
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        return ax


class BarycentricConstrainedColors(BarycentricColors):

    def col(self, x, y):
        from scipy.stats import norm
        b = self.cart2bary(x, y)
        w = norm(-0.25, 0.175).cdf(numpy.min(b, axis=1)).reshape((len(b), 1))
        b = w * b
        b[b > 1.0] = 1.0
        b[b < 0.0] = 0.0
        return numpy.array((self._cols * b.transpose()).transpose())
