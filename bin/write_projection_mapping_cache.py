#!/usr/bin/env python
import numpy
import logging
from white_matter.wm_recipe.region_mapper import RegionMapper

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

    def show_img(self, mask, zoom=True, sz_x=8, show_poles=True, convolve_var=None):
        nz = numpy.nonzero(mask)
        out_img = self.img(mask, convolve_var=convolve_var)
        if zoom:
            y1, y2 = nz[0].min(), nz[0].max() + 1
            x1, x2 = nz[1].min(), nz[1].max() + 1
            out_img = out_img[y1:y2, x1:x2, :]
        from matplotlib import pyplot as plt
        sz_y = sz_x * (y2 - y1) / (x2 - x1)
        fig = plt.figure(figsize=(sz_x, sz_y))
        ax = fig.add_axes([0, 0, 1, 1])
        plt.axis('off')
        ax.imshow(out_img, extent=(x1-0.5, x2-0.5, y2-0.5, y1-0.5))
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
        return ax


class BarycentricMaskMapper(BarycentricColors):

    def __init__(self, mask, interactive=True, contract=0.75, **kwargs):
        self.__tmp_x = []
        self.__tmp_y = []
        self.__kwargs = kwargs
        self._mask = mask
        if interactive:
            self.get_triangle(mask)
        else:
            self._find_corners(contract=contract)

    def _find_corners(self, contract=0.75):
        from scipy.spatial import distance
        nz = numpy.vstack(numpy.nonzero(self._mask)).transpose()
        cog = numpy.mean(nz, axis=0)
        D = numpy.vstack([distance.euclidean(_nz, cog) for _nz in nz])
        p1 = numpy.argmax(D)
        pD = distance.squareform(distance.pdist(nz))
        p1D = numpy.array([distance.euclidean(_nz, nz[p1])
                           for _nz in nz])
        X, Y = numpy.meshgrid(range(len(p1D)), range(len(p1D)))
        mx = numpy.argmax(p1D[X] + p1D[Y] + pD)
        p2 = numpy.mod(mx, pD.shape[1])
        p3 = mx / pD.shape[1]
        ret = nz[[p1, p2, p3], -1::-1]
        ret = cog[-1::-1] + contract * (ret - cog[-1::-1])
        super(BarycentricMaskMapper, self).__init__(ret[:, 0], ret[:, 1],
                                                    **self.__kwargs)
        self.show_img()

    def get_triangle(self, mask):
        from matplotlib import pyplot as plt
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        plt.axis('off')
        ax.imshow(mask)

        def onclick(event):
            x = event.xdata
            y = event.ydata
            if not numpy.isnan(x) and not numpy.isnan(y):
                self.__tmp_x.append(x)
                self.__tmp_y.append(y)
                ax.plot(x, y, 'ko')
            if len(self.__tmp_x) >= 3:
                super(BarycentricMaskMapper, self).__init__(numpy.array(self.__tmp_x),
                                                             numpy.array(self.__tmp_y),
                                                             **self.__kwargs)
                plt.close(fig)
                self.show_img()

        fig.canvas.mpl_connect('button_press_event', onclick)

    def show_img(self, mask=None, **kwargs):
        if mask is None:
            mask = self._mask
        return super(BarycentricMaskMapper, self).show_img(mask, **kwargs)


class GeneralProjectionMapper(object):

    def __init__(self, annotation_volume, mapper, structure_tree):
        self._mapper = mapper
        self._vol = annotation_volume
        self._tree = structure_tree
        self._make_reverse_lookup()
        self._res_cache = {}
        self._prepared_for = None
        self._used_hemisphere = 2

    def _three_d2flat_idx(self, idx):
        return idx[:, 0] * numpy.prod(self._mapper.REFERENCE_SHAPE[1:]) +\
               idx[:, 1] * self._mapper.REFERENCE_SHAPE[-1] + idx[:, 2]

    def _region_ids(self, regions, resolve_to_leaf=False):
        if not isinstance(regions, list) or isinstance(regions, numpy.ndarray):
            regions = [regions]
        r_struc = self._tree.get_structures_by_acronym(regions)
        r_ids = numpy.array([_x['id'] for _x in r_struc])
        def resolver(r_ids):
            rslvd = [resolver(_chldr) if len(_chldr) else _base
                     for _base, _chldr in
                     zip(r_ids, self._tree.child_ids(r_ids))]
            return numpy.hstack(rslvd)

        if resolve_to_leaf:
            return resolver(r_ids)
        return r_ids

    def _make_reverse_lookup(self):
        Y, X = numpy.meshgrid(xrange(self._mapper.view_lookup.shape[1]),
                              xrange(self._mapper.view_lookup.shape[0]))
        self._rv_lookup = dict([(idx, (x, y)) for x, y, idx in
                                zip(X.flat, Y.flat, self._mapper.view_lookup.flat)])

    def transform_points(self, x, y, z):
        xyz = numpy.vstack([x, y, z]).transpose()
        flt = self._three_d2flat_idx((xyz / 100).astype(int))
        flt2idx = dict([(v, i) for i, v in enumerate(flt)])
        hits = numpy.in1d(self._mapper.paths.flat, flt).reshape(self._mapper.paths.shape)
        idxx = numpy.nonzero(hits.sum(axis=1))[0]
        out_coords = numpy.NaN * numpy.ones((xyz.shape[0], 2))
        for idx in idxx:
            nz = self._rv_lookup[idx]
            src_flts = numpy.intersect1d(self._mapper.paths[idx], flt)
            for _flt in src_flts:
                out_coords[flt2idx[_flt], :] = [nz[0].mean(), nz[1].mean()]
        return out_coords

    def _make_volume_mask(self, idxx):
        return numpy.in1d(self._vol.flat, idxx).reshape(self._vol.shape)

    def make_volume_mask(self, regions):
        idxx = self._region_ids(regions, resolve_to_leaf=True)
        return self._make_volume_mask(idxx)

    def mask_result(self, regions, R, G, B):
        tgt_mask = self.make_volume_mask(regions)
        out_R, out_G, out_B = [numpy.NaN * numpy.ones_like(R) for _ in range(3)]
        out_R[tgt_mask] = R[tgt_mask]
        out_G[tgt_mask] = G[tgt_mask]
        out_B[tgt_mask] = B[tgt_mask]
        return out_R, out_G, out_B

    def mask_hemisphere(self, A, mask_val=False):
        if self._used_hemisphere == 2:
            if A.ndim == 3:
                A[:, :(A.shape[1] / 2), :] = mask_val
            else:
                A[:, :(A.shape[1] / 2)] = mask_val
        elif self._used_hemisphere == 1:
            if A.ndim == 3:
                A[:, (A.shape[1] / 2):, :] = mask_val
            else:
                A[:, (A.shape[1] / 2):] = mask_val

    @staticmethod
    def post_processing(IMG, log=False, exponent=2.0, normalize=1.25, per_pixel=True):
        if log:
            IMG = numpy.log10(IMG)
            extrema = (numpy.nanmin(IMG[~numpy.isinf(IMG)]),
                       numpy.nanmax(IMG[~numpy.isinf(IMG)]))
            IMG = (IMG - extrema[0]) / (extrema[1] - extrema[0])
            IMG[numpy.isinf(IMG)] = 0
        if exponent != 1.0:
            IMG = IMG ** exponent
        if normalize is not None:
            img_sum = numpy.nansum(IMG, axis=2)
            if per_pixel:
                IMG = normalize * IMG / img_sum.reshape(img_sum.shape + (1, ))
            else:
                fac = normalize * (img_sum > 0).sum() / img_sum.sum()
                IMG *= fac
        IMG[IMG > 1.0] = 1.0
        return IMG

    def _emergency_swap_hemispheres(self):
        self._used_hemisphere = 1
        self._mask2d = self._mapper.transform(self._mask3d)
        self.mask_hemisphere(self._mask2d)
        self._bary = BarycentricMaskMapper(self._mask2d, interactive=False)

    def prepare_for_source(self, src, interactive=True, contract=0.75):
        if src == self._prepared_for:
            return
        else:
            self._used_hemisphere = 2
            self._res_cache = {}
            self._prepared_for = src
            self._mask3d = self.make_volume_mask(src)
            self._mask2d = self._mapper.transform(self._mask3d)
            self.mask_hemisphere(self._mask2d)
            self._bary = BarycentricMaskMapper(self._mask2d, interactive=interactive,
                                               contract=contract)

    def _for_full_volume(self, **kwargs):
        raise NotImplementedError("This is implemented in derived classes")

    def for_target(self, tgt, **kwargs):
        kk = sorted(kwargs.keys())
        signature = tuple([kwargs[_k] for _k in kk])
        if signature not in self._res_cache:
            self._res_cache[signature] = self._for_full_volume(**kwargs)
        out_R, out_G, out_B = self.mask_result(tgt, *self._res_cache[signature])
        return numpy.dstack([self._mapper.transform(out_R, agg_func=numpy.nanmean),
                             self._mapper.transform(out_G, agg_func=numpy.nanmean),
                             self._mapper.transform(out_B, agg_func=numpy.nanmean)])

    def draw_source(self, ax=None, **kwargs):
        from matplotlib import pyplot as plt
        if ax is None:
            ax = plt.figure().add_axes([0, 0, 1, 1])
            plt.axis('off')
        ax.contour(self._mask2d, **kwargs)

    def _imshow(self, IMG, ax=None):
        from matplotlib import pyplot as plt
        if ax is None:
            ax = plt.figure(figsize=(16, 8)).add_axes([0, 0, 1, 1])
            plt.axis('off')
        ax.imshow(IMG)
        return ax

    def draw_projection(self, tgt, ax=None, target_args={}, pp_args={},
                        src_args={}, draw_source=True, return_img=False):
        IMG = self.for_target(tgt, **target_args)
        IMG = self.post_processing(IMG, **pp_args)
        ax = self._imshow(IMG, ax=ax)
        if draw_source:
            self.draw_source(ax=ax, **src_args)
        if return_img:
            return ax, IMG
        return ax

    def _fit_func(self, abc, xy):
        from scipy import optimize
        initial_solution = numpy.linalg.lstsq(abc, xy, rcond=-1)[0]

        def evaluate_error(pt):
            res = BarycentricCoordinates(pt[:3], pt[3:])
            return (res.cart2bary(xy[:, 0], xy[:, 1]) - abc).flatten()

        info_log.info("\tMean initial error is %f" %
                      numpy.abs(evaluate_error(initial_solution.transpose().flatten())).mean())
        sol = optimize.leastsq(evaluate_error, initial_solution.transpose().flatten(),
                               full_output=True)
        final_error = numpy.abs(sol[2]['fvec']).mean()
        info_log.info("\tMean final error is %f" % final_error)
        from .contract import contract, estimate_mapping_var
        x_out, y_out, map_var = contract(sol[0][:3], sol[0][3:], xy, #numpy.sqrt(1/numpy.mean(abc, axis=0)),
                                         info_log)
        col_sys = BarycentricColors(x_out, y_out)
        map_var = estimate_mapping_var(abc, col_sys.col(xy[:, 0], xy[:, 1]))
        #x_out, y_out, map_var = self._contract(sol[0][:3], sol[0][3:], xy)
        return col_sys, numpy.maximum(map_var, 0.25), final_error

    def fit_target_coordinates(self, IMG):
        self.mask_hemisphere(IMG, mask_val=numpy.NaN)
        valid = numpy.all(~numpy.isnan(IMG), axis=2) & (numpy.nansum(IMG, axis=2) > 0)
        nz = numpy.nonzero(valid)
        abc = IMG[nz[0], nz[1], :]
        abc = abc / abc.sum(axis=1).reshape((abc.shape[0], 1))
        xy = numpy.vstack([nz[1], nz[0]]).transpose()
        return self._fit_func(abc, xy)

    def make_target_region_coordinate_system(self, tgt, target_args={}, pp_use={},
                                             pp_display={},
                                             src_args={}, draw=True):
        if draw:
            ax, IMG = self.draw_projection(tgt, target_args=target_args,
                                           pp_args=pp_display, src_args=src_args,
                                           draw_source=True,
                                           return_img=True)
        IMG = self.for_target(tgt, **target_args)
        IMG = self.post_processing(IMG, **pp_use)
        info_log.info("Fitting for target %s" % tgt)
        res_coords, map_var, final_error = self.fit_target_coordinates(IMG)
        if numpy.isnan(map_var): map_var = 0.25
        info_log.info("Mapping variance is: %f" % map_var)
        if draw:
            info_log.info("Drawing results\n\n")
            tgt_mask = self._mapper.transform(self.make_volume_mask(tgt))
            self.mask_hemisphere(tgt_mask)
            ax2 = res_coords.show_img(tgt_mask, convolve_var=map_var)
            return res_coords, map_var, final_error, ax, ax2
        return res_coords, map_var, final_error


class VoxelArrayBaryMapper(GeneralProjectionMapper):

    def __init__(self, voxel_array, source_coords, target_coords,
                 annotation_volume, mapper, structure_tree):
        super(VoxelArrayBaryMapper, self).__init__(annotation_volume, mapper, structure_tree)
        self._voxel_array = voxel_array
        self._source_coords = source_coords
        self._target_coords = target_coords
        self._source_flat = self._three_d2flat_idx(self._source_coords)
        self._source_dict = dict([(v, i) for i, v in enumerate(self._source_flat)])

    @classmethod
    def from_cache(cls, cache):
        from mcmodels.core.cortical_map import CorticalMap
        voxel_array, source_mask, target_mask = cache.get_voxel_connectivity_array()
        vol, _ = cache.get_annotation_volume()
        M = CorticalMap(projection='dorsal_flatmap')
        tree = cache.get_structure_tree()
        return cls(voxel_array, source_mask.coordinates, target_mask.coordinates,
                   vol, M, tree)

    def proj_for_voxel_3d(self, idx):
        idx = self._source_dict[idx]
        row = self._voxel_array[idx, :]
        vol = numpy.zeros(self._mapper.REFERENCE_SHAPE)
        vol[self._target_coords[:, 0], self._target_coords[:, 1],
            self._target_coords[:, 2]] = row
        return vol

    def proj_for_voxel_2d(self, idx):
        return self._mapper.transform(self.proj_for_voxel_3d(idx), agg_func=numpy.nanmean)

    def cols_and_paths(self, mask2d, bary, shuffle=False):
        nz2d = numpy.nonzero(mask2d)
        cols = bary.col(nz2d[1], nz2d[0])
        paths = self._mapper.paths[self._mapper.view_lookup[mask2d]]
        return cols, paths, zip(*nz2d)

    def paths_in_source_volume(self, mask3d):
        nz3d = numpy.nonzero(mask3d.flat)[0]
        return nz3d[numpy.in1d(nz3d, self._source_flat)]

    def prepare_for_source(self, src, interactive=True, contract=0.75):
        super(VoxelArrayBaryMapper, self).prepare_for_source(src, interactive=interactive,
                                                             contract=contract)
        self._relevant_paths = self.paths_in_source_volume(self._mask3d)

    def _for_full_volume(self, thresh=1.5, shuffle=False, normalize=True):
        import progressbar
        self._cols, self._paths, smpl_pts = self.cols_and_paths(self._mask2d, self._bary, shuffle=shuffle)
        self._paths = [numpy.intersect1d(_path, self._relevant_paths)
                       for _path in self._paths]
        value = numpy.abs(numpy.diff(numpy.hstack([self._cols, self._cols[:, 0:1]]), axis=1)).sum(axis=1)
        out_R = numpy.zeros(self._mask3d.shape, dtype=float)
        out_G = numpy.zeros(self._mask3d.shape, dtype=float)
        out_B = numpy.zeros(self._mask3d.shape, dtype=float)
        out_count = numpy.zeros(self._mask3d.shape, dtype=float)
        pbar = progressbar.ProgressBar()
        self._sampled = []

        for _col, _path, _val, _pt in pbar(zip(self._cols, self._paths, value, smpl_pts)):
            if _val > thresh and len(_path) > 0:
                proj = numpy.stack([self.proj_for_voxel_3d(_p) for _p in _path],
                                   axis=3).mean(axis=3)
                out_count += proj
                if shuffle:
                    _col = numpy.random.permutation(_col)
                out_R += _col[0] * proj
                out_G += _col[1] * proj
                out_B += _col[2] * proj
                self._sampled.append((_pt[0], _pt[1], _col))

        if normalize:
            out_R /= out_count; out_G /= out_count; out_B /= out_count
        return out_R, out_G, out_B

    def draw_projection(self, *args, **kwargs):
        ret = super(VoxelArrayBaryMapper, self).draw_projection(*args, **kwargs)
        if kwargs.get('return_img', False):
            ax, IMG = ret
        else:
            ax = ret
        for _y, _x, _c in self._sampled:
            ax.plot(_x, _y, 'o', color=_c)
        return ret


class VoxelNodeBaryMapper(GeneralProjectionMapper):

    def __init__(self, cache, annotation_volume, mapper, structure_tree):
        super(VoxelNodeBaryMapper, self).__init__(annotation_volume, mapper, structure_tree)
        self._cache = cache

    @classmethod
    def from_cache(cls, cache):
        from mcmodels.core.cortical_map import CorticalMap
        vol, _ = cache.get_annotation_volume()
        M = CorticalMap(projection='dorsal_flatmap')
        tree = cache.get_structure_tree()
        return cls(cache, vol, M, tree)

    def project_experiments(self, experiments):
        s = ['injection_x', 'injection_y', 'injection_z']
        xyz = numpy.vstack([[_e[_s] for _s in s] for _e in experiments])
        return self.transform_points(xyz[:, 0], xyz[:, 1], xyz[:, 2])

    def prepare_for_source(self, src, interactive=True, contract=0.75, cre=False):
        super(VoxelNodeBaryMapper, self).prepare_for_source(src, interactive=interactive,
                                                            contract=contract)
        idx_inj = self._region_ids(src)
        self._exp = self._cache.get_experiments(cre=cre, injection_structure_ids=idx_inj)
        self._exp_locs = self.project_experiments(self._exp)

        tmp_y = self._exp_locs[:, 1]
        if numpy.mean(tmp_y[~numpy.isnan(tmp_y)] < (self._mask2d.shape[1] / 2)) > 0.5:
            info_log.info("Exceptionally using LEFT hemisphere!")
            self._emergency_swap_hemispheres()

    def assign_colors(self, exp_locs, shuffle=False):
        out_cols = numpy.vstack([self._bary.col(_p[1], _p[0]) for _p in exp_locs])
        if self._used_hemisphere == 2:
            out_cols[exp_locs[:, 1] < self._mask2d.shape[1]/2, :] = numpy.NaN
        else:
            out_cols[exp_locs[:, 1] > self._mask2d.shape[1]/2, :] = numpy.NaN
        if shuffle:
            for i in range(len(out_cols)):
                out_cols[i, :] = numpy.random.permutation(out_cols[i, :])
            out_cols = out_cols[numpy.random.permutation(len(out_cols))]
        return out_cols

    def _summation(self, proj, cols, normalize_bias=True,
                  normalize_pixels=True):
        import progressbar
        t_shape = self._mapper.REFERENCE_SHAPE
        out_R = numpy.zeros(t_shape, dtype=float)
        out_G = numpy.zeros(t_shape, dtype=float)
        out_B = numpy.zeros(t_shape, dtype=float)
        out_count = numpy.zeros(t_shape, dtype=float)
        pbar = progressbar.ProgressBar()

        for _col, _p in pbar(zip(cols, proj)):
            if not numpy.any(numpy.isnan(_col)):
                out_count += _p[0]
                out_R += _col[0] * _p[0]
                out_G += _col[1] * _p[0]
                out_B += _col[2] * _p[0]
        if normalize_bias:
            out_R /= (numpy.nanmean(cols[:, 0]) + 1E-3)
            out_G /= (numpy.nanmean(cols[:, 1]) + 1E-3)
            out_B /= (numpy.nanmean(cols[:, 2]) + 1E-3)
        if normalize_pixels:
            out_R /= out_count
            out_G /= out_count
            out_B /= out_count
        return (out_R, out_G, out_B)

    def _for_full_volume(self, shuffle=False, **kwargs):
        if shuffle:
            raise Exception("shuffle disabled for now!")
        self._exp_cols = self.assign_colors(self._exp_locs, shuffle=shuffle)
        proj = [self._cache.get_projection_density(_e['id'])
                for _e in self._exp]
        return self._summation(proj, self._exp_cols, **kwargs)

    def draw_projection(self, *args, **kwargs):
        draw_experiments = kwargs.pop('draw_experiments', True)
        ret = super(VoxelNodeBaryMapper, self).draw_projection(*args, **kwargs)
        if kwargs.get('return_img', False):
            ax, IMG = ret
        else:
            ax = ret
        if draw_experiments:
            for _e, _c in zip(self._exp_locs, self._exp_cols):
                ax.plot(_e[1], _e[0], 'o', color=_c)
        return ret


def make_mapper(cfg):
    import mcmodels
    if cfg['class'] == 'VoxelNodeBaryMapper':
        cls = VoxelNodeBaryMapper
    elif cfg['class'] == 'VoxelArrayBaryMapper':
        cls = VoxelArrayBaryMapper
    else:
        raise Exception("Unknown mapper class: %s" % cfg['class'])
    cache = mcmodels.core.VoxelModelCache(
        manifest_file=cfg["cache_manifest"])
    obj = cls.from_cache(cache)
    return obj


def main(cfg, obj, src):
    import h5py
    import os
    from matplotlib import pyplot as plt
    mpr = RegionMapper()

    cfg_root = cfg["cfg_root"]

    target_args = cfg["target_args"]
    pp_use = cfg["pp_use"]
    pp_display = cfg["pp_display"]
    prepare_args = cfg["prepare_args"]
    if "cre" in prepare_args and prepare_args["cre"] == "None":
        print "Using both cre positive and negative experiments"
        prepare_args["cre"] = None

    out_plots = cfg["plot_dir"]
    if not os.path.isabs(out_plots):
        out_plots = os.path.join(cfg_root, out_plots)
    if not os.path.exists(out_plots):
        os.makedirs(out_plots)

    out_h5_fn = cfg["h5_fn"]
    if not os.path.isabs(out_h5_fn):
        out_h5_fn = os.path.join(cfg_root, out_h5_fn)
    if not os.path.exists(os.path.split(out_h5_fn)[0]):
        os.makedirs(os.path.split(out_h5_fn)[0])
    if os.path.exists(out_h5_fn):
        h5 = h5py.File(out_h5_fn, 'r+')
    else:
        h5 = h5py.File(out_h5_fn, 'w')

    obj.prepare_for_source(src, interactive=False, **prepare_args)

    grp = h5.require_group(str(src))
    grp_coords = grp.require_group('coordinates')
    grp_coords.attrs['base_coord_system'] = "Allen Dorsal Flatmap"
    grp_coords.require_dataset('x', (3,), float, data=obj._bary._x)
    grp_coords.require_dataset('y', (3,), float, data=obj._bary._y)
    grp = grp.require_group('targets')

    for tgt in mpr.region_names:
        try:
            tgt_grp = grp.require_group(tgt)
            tgt_plots = os.path.join(out_plots, str(src), str(tgt))
            if not os.path.exists(tgt_plots):
                os.makedirs(tgt_plots)
            if 'coordinates/base_coord_system' in tgt_grp:
                info_log.info("%s/%s already present. Skipping..." %
                              (str(src), str(tgt)))
                continue
            res, map_var, error, ax1, ax2 = obj.make_target_region_coordinate_system(tgt,
                                    target_args=target_args, pp_use=pp_use,
                                    pp_display=pp_display, draw=True)
            ax1.figure.savefig(os.path.join(tgt_plots, ('%s_data' % str(tgt)) + cfg["plot_extension"]))
            ax2.figure.savefig(os.path.join(tgt_plots, ('%s_model' % str(tgt)) + cfg["plot_extension"]))
            plt.close('all')
            tgt_grp.create_dataset('coordinates/base_coord_system', data="Allen Dorsal Flatmap")
            tgt_grp.create_dataset('coordinates/x', data=res._x)
            tgt_grp.create_dataset('coordinates/y', data=res._y)
            tgt_grp.create_dataset('mapping_variance', data=[map_var])
            tgt_grp.create_dataset('error', data=[error])
            if isinstance(obj, VoxelNodeBaryMapper):
                N = numpy.all(~numpy.isnan(obj._exp_cols), axis=1).sum()
                tgt_grp.create_dataset('n_experiments', data=[N])
            h5.flush()
        except Exception as exc:
            print "Trouble with %s/%s" % (str(src), str(tgt))
            continue
    h5.close()


if __name__ == "__main__":
    import sys
    import json
    import os
    cfg_file = sys.argv[1]
    with open(cfg_file, 'r') as fid:
        cfg = json.load(fid)["ProjectionMapping"]
    cfg["cfg_root"] = os.path.split(cfg_file)[0]
    obj = make_mapper(cfg)
    if len(sys.argv) > 2:
        main(cfg, obj, sys.argv[2])
    else:
        M = RegionMapper()
        for src in M.region_names:
            main(cfg, obj, src)
