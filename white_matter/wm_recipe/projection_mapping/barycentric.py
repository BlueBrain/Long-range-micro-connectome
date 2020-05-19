import numpy
import logging

logging.basicConfig(level=1)
info_log = logging.getLogger(__file__)


class BarycentricCoordinates(object):

    def __init__(self, *args):
        """
        :param args: x, y and z coordinates of three points, i.e. cls(x, y, z)
        or x and y coordinates in the two-dimensional case, i.e. cls(x, y)
        """
        assert numpy.all(numpy.array(list(map(len, args))) == 3)
        self._dim = len(args)
        assert self._dim >= 2 and self._dim <= 3
        self._coords = args
        self._S = numpy.matrix(self._coords).transpose()
        self._T = numpy.matrix(self.transform_matrix())

    def transform_matrix(self):
        T = numpy.array([[_c[0] - _c[2], _c[1] - _c[2]]
                         for _c in self._coords])
        # In 3-d case we append the normal of the defined plane. In conversions we will ignore this third coordinate
        if self._dim == 3:
            N = numpy.cross(T[:, 0], T[:, 1])
            N = N / numpy.linalg.norm(N)
            T = numpy.hstack([T, numpy.vstack(N)])
        return T

    def cart2bary(self, *args):
        assert len(args) == self._dim
        lc = [c - b[2] for c, b in zip(args, self._coords)]
        res = numpy.linalg.solve(self._T, numpy.vstack(lc))
        # If it's the 3-d case this is where we ignore the normal component
        return numpy.vstack([res[0, :], res[1, :], 1.0 - res[:2, :].sum(axis=0)]).transpose()

    def bary2cart(self, a, b, c):
        return numpy.array(numpy.vstack([a, b, c]).transpose() * self._S)

    def area(self):
        p1 = numpy.array([_c[0] - _c[2] for _c in self._coords])
        p2 = numpy.array([_c[1] - _c[2] for _c in self._coords])
        l1 = numpy.linalg.norm(p1)
        l2 = numpy.linalg.norm(p2)
        assert l1 > 0 and l2 > 0
        return numpy.sin(numpy.arccos(numpy.sum(p1 * p2) / (l1 * l2))) * (l1 * l2) * 0.5


class BarycentricFlatmap(BarycentricCoordinates):
    """A barycentric coordinate system that in the 3d case additionally provides the implied flatmap.
    That is, the flatmap defined as follows: For a 3d point, first parallel-project it into the plane
    of the barycentric triangle. Then convert it to the orthonormal base given by its first base vector and
    an orthogonal vector in the plane."""
    def __init__(self, *args):
        super(BarycentricFlatmap, self).__init__(*args)
        if self._dim == 3:
            self._initialize_flatmap()
            self.implied_flatmap = self.__implied_flatmap_3d
        else:
            self.implied_flatmap = self.__implied_flatmap_2d

    def _initialize_flatmap(self):
        pO = [_c[2] for _c in self._coords]
        T = numpy.array(self._T)
        v1 = T[:, 0]
        v2 = T[:, 1]
        v1 = v1 / numpy.linalg.norm(v1)
        N = numpy.cross(v1, v2)
        N = N / numpy.linalg.norm(N)
        v2 = numpy.cross(v1, N)
        self._flatmapper = BarycentricCoordinates([pO[0] + v1[0], pO[0] + v2[0], pO[0]],
                                                  [pO[1] + v1[1], pO[1] + v2[1], pO[1]],
                                                  [pO[2] + v1[2], pO[2] + v2[2], pO[2]])

    def __implied_flatmap_3d(self, *args):
        out = self._flatmapper.cart2bary(*args)
        return out[:, :2]

    def __implied_flatmap_2d(self, *args):
        return numpy.vstack(args).transpose()


class BarycentricColors(BarycentricFlatmap):

    # noinspection PyDefaultArgument
    def __init__(self, *args, **kwargs):
        super(BarycentricColors, self).__init__(*args)
        self._cols = numpy.matrix(numpy.vstack([kwargs.get('red', [1, 0, 0]),
                                                kwargs.get('green', [0, 1, 0]),
                                                kwargs.get('blue', [0, 0, 1])]).transpose())

    def col(self, *args):
        b = self.cart2bary(*args)
        b[b > 1.0] = 1.0
        b[b < 0.0] = 0.0
        return numpy.array((self._cols * b.transpose()).transpose())

    def img(self, mask, convolve_var=None):
        nz = numpy.nonzero(mask)
        out_img = numpy.zeros(mask.shape + (3,))
        out_img[nz] = self.col(*[_nz.astype(float) for _nz in nz])
        if self._dim < 3:
            if convolve_var is not None: # No 3d convolution...
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
        return out_img # numpy.transpose(out_img, [1, 0, 2])

    def project_img(self, img, aggregrate_func=numpy.nanmean):
        nz = numpy.nonzero(numpy.nansum(img, axis=-1))
        xy = self.implied_flatmap(*[_nz.astype(float) for _nz in nz])
        corners = numpy.array([map(int, format(i, "03b")) for i in range(8)])
        corners = corners * numpy.array(img.shape[:-1])
        corners = self.implied_flatmap(*corners.transpose()).astype(int)
        fr = corners.min(axis=0) - 1
        to = corners.max(axis=0) + 2
        bins = [numpy.arange(a,b) for a,b in zip(fr, to)] # x, y

        img_out = numpy.zeros((len(bins[0]), len(bins[1]), 3)) # x, y
        img = img[nz]
        x, y = [numpy.digitize(_coords, bins=_bins) for _coords, _bins
                in zip(xy.transpose(), bins)] # x, y
        for _x in numpy.unique(x):
            for _y in numpy.unique(y):
                vals = img[(x == _x) & (y == _y)]
                img_out[_x, _y] = aggregrate_func(vals, axis=0) # x, y
        extent = (fr[0], to[0], to[1], fr[1]) # xmin, xmax, ymin, ymax
        return img_out, extent


    # noinspection PyTypeChecker
    def show_img(self, mask, zoom=True, sz_x=8, show_poles=True, convolve_var=None):
        nz = numpy.nonzero(mask)
        out_img = self.img(mask, convolve_var=convolve_var)
        poles = self.implied_flatmap(*map(numpy.array, self._coords))

        if self._dim == 3:
            out_img, extent = self.project_img(out_img)
            x1, x2, y2, y1 = extent
        else:
            if zoom:
                y1, y2 = nz[1].min(), nz[1].max() + 1
                x1, x2 = nz[0].min(), nz[0].max() + 1
                out_img = out_img[x1:x2, y1:y2, :]
            else:
                x1 = 0; x2 = out_img.shape[0]
                y1 = 0; y2 = out_img.shape[1]
        from matplotlib import pyplot as plt
        sz_y = sz_x * (y2 - y1) / (x2 - x1)
        fig = plt.figure(figsize=(sz_x, sz_y))
        ax = fig.add_axes([0, 0, 1, 1])
        plt.axis('off')
        ax.imshow(out_img,
                  extent=(y1-0.5, y2-0.5, x2-0.5, x1-0.5))
        #ax.contour(mask[y1:y2, x1:x2], 1,
        #           extent=(x1-0.5, x2-0.5, y1-0.5, y2-0.5),
        #           colors=['w'])
        if show_poles:
            for pt, col in zip(poles, numpy.array(self._cols)):
                ax.plot(pt[1], pt[0], 'v',
                        color=[1.0, 1.0, 1.0], markersize=15)
                ax.plot(pt[1], pt[0], 'v',
                        color=col,
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

    def col(self, *args):
        from scipy.stats import norm
        b = self.cart2bary(*args)
        w = norm(-0.25, 0.175).cdf(numpy.min(b, axis=1)).reshape((len(b), 1))
        b = w * b
        b[b > 1.0] = 1.0
        b[b < 0.0] = 0.0
        return numpy.array((self._cols * b.transpose()).transpose())
