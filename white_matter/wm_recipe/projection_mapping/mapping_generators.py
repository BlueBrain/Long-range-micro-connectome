import numpy
import logging
from matplotlib import pyplot as plt
from .barycentric import BarycentricConstrainedColors, BarycentricColors
from .contract import contract_min
from .custom_flatmap import NrrdFlatmap


logging.basicConfig(level=1)
info_log = logging.getLogger(__file__)


# noinspection PyMissingConstructor
class BarycentricMaskMapper(BarycentricColors):

    def __init__(self, mask, interactive=True, contract=0.75, **kwargs):
        self.__tmp_x = []
        self.__tmp_y = []
        self.__kwargs = kwargs
        self._mask = mask
        if mask is not None:
            self._nz = numpy.vstack(numpy.nonzero(self._mask)).transpose()
        self._dim = self._nz.shape[1]
        if interactive:
            self.get_triangle()
        else:
            self._find_corners(contract=contract)

    def _find_corners(self, contract=0.75):
        from scipy.spatial import distance, distance_matrix
        nz = self._nz
        cog = numpy.mean(nz, axis=0)

        D = distance_matrix([cog], nz)[0]
        p1 = numpy.argmax(D) # first point: Furthest away from center of region
        pD = distance.squareform(distance.pdist(nz)) # pairwise distance for all pairs
        p1D = distance_matrix(nz, nz[p1:(p1+1)]) # distance to first point. Shape: N x 1

        mx = numpy.argmax(pD + p1D + p1D.transpose()) # maximize sum of distances between the three points
        p2 = int(numpy.mod(mx, pD.shape[1]))
        p3 = int(mx / pD.shape[1])
        ret = nz[[p1, p2, p3]]
        ret = cog + contract * (ret - cog)
        super(BarycentricMaskMapper, self).__init__(ret[:, 0], ret[:, 1],
                                                    **self.__kwargs)
        self.show_img()

    def _image_figure(self):
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        plt.axis('off')
        ax.imshow(self._mask)
        return fig, ax

    def get_triangle(self):
        fig, ax = self._image_figure()

        def onclick(event):
            y = event.ydata
            x = event.xdata
            print(x, y)
            if not numpy.isnan(x) and not numpy.isnan(y):
                # x,y flipped because of behavior of imshow
                self.__tmp_x.append(y)
                self.__tmp_y.append(x)
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


class IrregularGridMapper(BarycentricMaskMapper):

    def __init__(self, xy, interactive=True, contract=0.75, **kwargs):
        self._nz = xy
        super(IrregularGridMapper, self).__init__(None, interactive=interactive, contract=contract)

    def _image_figure(self):
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        plt.axis('off')
        for x, y in self._nz:
            # x, y flipped to be consistent with imshow in the base class
            ax.plot(y, x, 's', ms=20, color='black')
        return fig, ax

    def show_img(self, mask=None, **kwargs):
        if mask is not None:
            return super(BarycentricMaskMapper, self).show_img(mask, **kwargs)
        cols = self.col(self._nz[:, 0], self._nz[:, 1])
        ax = plt.figure().gca()
        for x, y, col in zip(self._nz[:, 0], self._nz[:, 1], cols):
            ax.plot(x, y, 's', ms=20, color=col)
        return ax


# noinspection PyUnresolvedReferences,PyDefaultArgument
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
        return idx[:, 0] * numpy.prod(self._mapper.REFERENCE_SHAPE[1:]) + \
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
        Y, X = numpy.meshgrid(range(self._mapper.view_lookup.shape[1]),
                              range(self._mapper.view_lookup.shape[0]))
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

    def mask_hemisphere(self, A, flatmap=True, mask_val=False):
        if self._used_hemisphere == 2:
            if not flatmap:
                A[:, :, :int(A.shape[2] / 2)] = mask_val
            else:
                A[:, :int(A.shape[1] / 2)] = mask_val
        elif self._used_hemisphere == 1:
            if not flatmap:
                A[:, :, int(A.shape[2] / 2):] = mask_val
            else:
                A[:, int(A.shape[1] / 2):] = mask_val

    @staticmethod
    def post_processing(IMG, log=False, exponent=None, normalize=None, per_pixel=True,
                        relative_cutoff=None, equalize=(0.1, 0.5)):
        if log:
            IMG = numpy.log10(IMG)
            extrema = (numpy.nanmin(IMG[~numpy.isinf(IMG)]),
                       numpy.nanmax(IMG[~numpy.isinf(IMG)]))
            IMG = (IMG - extrema[0]) / (extrema[1] - extrema[0])
            IMG[numpy.isinf(IMG)] = 0
        if exponent is not None:
            IMG = IMG ** exponent
        if equalize is not None:
            channels = IMG[IMG.sum(axis=-1) > 0]
            if numpy.min(channels.mean(axis=0) / channels.mean(axis=0).sum()) < equalize[0]:
                tgt = channels.mean(axis=0) ** equalize[1]
                facs = (tgt * channels.mean(axis=0).sum()) / (tgt.sum() * channels.mean(axis=0))
                IMG = IMG * facs.reshape(tuple(numpy.ones(IMG.ndim - 1, dtype=int)) + (3,))
        if relative_cutoff is not None:
            IMGsum = numpy.sum(IMG, axis=-1)
            cutoff = numpy.percentile(IMGsum[~numpy.isnan(IMGsum) & (IMGsum > 0)],
                                      relative_cutoff[0]) * relative_cutoff[1]
            IMG = IMG / cutoff
            over = numpy.nonzero(IMG.sum(axis=-1) > 1.0)
            IMG[over] = IMG[over] / IMG[over].sum(axis=1).reshape((len(over[0]), 1))
        if normalize is not None:
            img_sum = numpy.nansum(IMG, axis=2)
            if per_pixel:
                IMG = normalize * IMG / img_sum.reshape(img_sum.shape + (1,))
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

    def for_target(self, tgt, flatmap=True, **kwargs):
        kk = sorted(kwargs.keys())
        signature = tuple([kwargs[_k] for _k in kk])
        if signature not in self._res_cache:
            self._res_cache[signature] = self._for_full_volume(**kwargs)
        out_R, out_G, out_B = self.mask_result(tgt, *self._res_cache[signature])
        if flatmap:
            return numpy.dstack([self._mapper.transform(out_R, agg_func=numpy.nanmean),
                                 self._mapper.transform(out_G, agg_func=numpy.nanmean),
                                 self._mapper.transform(out_B, agg_func=numpy.nanmean)])
        else:
            return numpy.stack([out_R, out_G, out_B], 3)

    def draw_source(self, ax=None, **kwargs):
        from matplotlib import pyplot as plt
        if ax is None:
            ax = plt.figure().add_axes([0, 0, 1, 1])
            plt.axis('off')
        ax.contour(self._mask2d, **kwargs)

    @staticmethod
    def _imshow(IMG, ax=None):
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

    @staticmethod
    def _mapping_overlap(bary_src, coord_sys, xy, min_dist=3.0):
        from scipy.spatial import distance_matrix
        cart_src = coord_sys.bary2cart(bary_src[:, 0], bary_src[:, 1], bary_src[:, 2])
        H = distance_matrix(xy, cart_src)
        coverage_src = (H.min(axis=0) <= min_dist).mean()
        coverage_tgt = (H.min(axis=1) <= min_dist).mean()
        return coverage_src, coverage_tgt

    def mapping_overlap(self, coord_sys, xy, min_dist=0.05):
        src_x, src_y = numpy.nonzero(self._mask2d)
        bary_src = self._bary.cart2bary(src_x, src_y)
        return self._mapping_overlap(bary_src, coord_sys, xy, min_dist=min_dist)

    def _fit_func_permutation_only(self, abc, xy, *args, **kwargs):
        from white_matter.wm_recipe.projection_mapping.contract import estimate_mapping_var

        candidate = IrregularGridMapper(xy, interactive=False)
        A = candidate.col(*xy.transpose())
        permutations = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
        errors = [numpy.abs(A[:, permutation] - abc).mean() for permutation in permutations]
        idxx = numpy.argmin(errors)
        solution = [coords[permutations[idxx]] for coords in candidate._coords]

        col_sys = BarycentricConstrainedColors(*solution)
        map_var = estimate_mapping_var(abc, col_sys.col(*xy.transpose()))
        map_var = numpy.maximum(map_var, 0.25)
        overlaps = self.mapping_overlap(col_sys, xy, min_dist=map_var)

        return col_sys, map_var, overlaps, errors[idxx]

    def _fit_func(self, abc, xy, exponent=1.0, mul_angle=10.0, mul_overlap=10.0, opt_args={}):
        def _max_angle(pt):
            from scipy.stats import norm
            A = pt.reshape(-1, 3).transpose()
            normalize = lambda _x: _x / numpy.linalg.norm(_x)
            cos_ang = [normalize(A[1] - A[0]).dot(normalize(A[2] - A[0])),
                       normalize(A[0] - A[1]).dot(normalize(A[2] - A[1])),
                       normalize(A[0] - A[2]).dot(normalize(A[1] - A[2]))]
            return mul_angle * (norm(2.95, 0.218).cdf(numpy.arccos(cos_ang).max())
                                + 1 - norm(0.174, 0.2).cdf(numpy.arccos(cos_ang).min())) + 1

        def _overlap(cols):
            diff = numpy.max(cols, axis=0) - numpy.min(cols, axis=0)
            coverage = numpy.prod(diff)
            return mul_overlap * (1 - coverage ** exponent) + 1

        from scipy import optimize
        v = abc.sum(axis=1) > 0.9
        initial_solution = numpy.linalg.lstsq(abc[v], xy[v], rcond=-1)[0]

        def evaluate_error(pt):
            res = BarycentricConstrainedColors(*numpy.split(pt, numpy.arange(3, len(pt), 3)))
            cols = res.col(*xy.transpose())
            return (numpy.abs(cols - abc)).flatten() * _overlap(cols) * _max_angle(pt)

        info_log.info("\tInitial solution: %s" % str(initial_solution))
        info_log.info("\tMean initial error is %f" %
                      numpy.abs(evaluate_error(initial_solution.transpose().flatten())).mean())
        sol = optimize.leastsq(evaluate_error, initial_solution.transpose().flatten(),
                               full_output=True, **opt_args)
        final_error = numpy.abs(sol[2]['fvec']).mean()
        info_log.info("\tMean final error is %f" % final_error)
        from white_matter.wm_recipe.projection_mapping.contract import estimate_mapping_var

        params_out = contract_min(sol[0], xy)
        col_sys = BarycentricConstrainedColors(*params_out)
        map_var = estimate_mapping_var(abc, col_sys.col(*xy.transpose()))
        map_var = numpy.maximum(map_var, 0.25)
        overlaps = self.mapping_overlap(col_sys, xy, min_dist=map_var)

        return col_sys, map_var, overlaps, final_error

    def fit_target_coordinates(self, IMG, **kwargs):
        valid = numpy.all(~numpy.isnan(IMG), axis=-1) & (numpy.nansum(IMG, axis=-1) > 0)
        nz = numpy.nonzero(valid)
        abc = IMG[nz]
        xy = numpy.vstack(nz).transpose()
        if bool(kwargs.pop("only_permutation", False)):
            return self._fit_func_permutation_only(abc, xy, **kwargs)
        return self._fit_func(abc, xy, **kwargs)

    # noinspection PyUnboundLocalVariable
    def make_target_region_coordinate_system(self, tgt, target_args={}, pp_use={},
                                             pp_display={}, fit_args={},
                                             src_args={}, flatmap=True, draw=True):
        if draw:
            ax, IMG = self.draw_projection(tgt, target_args=target_args,
                                           pp_args=pp_display, src_args=src_args,
                                           draw_source=True,
                                           return_img=True)
        IMG = self.for_target(tgt, flatmap=flatmap, **target_args)
        self.mask_hemisphere(IMG, flatmap=flatmap, mask_val=numpy.NaN)
        IMG = self.post_processing(IMG, **pp_use)
        info_log.info("Fitting for target %s" % tgt)
        res_coords, map_var, overlaps, final_error = self.fit_target_coordinates(IMG, **fit_args)
        if numpy.isnan(map_var): map_var = 0.25
        info_log.info("Mapping variance is: %f" % map_var)
        info_log.info("Source and target region overlap: %s" % str(overlaps))
        if draw:
            info_log.info("Drawing results\n\n")
            tgt_mask = self._mapper.transform(self.make_volume_mask(tgt))
            self.mask_hemisphere(tgt_mask)
            ax2 = res_coords.show_img(tgt_mask, convolve_var=map_var)
            return res_coords, map_var, overlaps, final_error, ax, ax2
        return res_coords, map_var, overlaps, final_error


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
    def from_cache(cls, cache, custom_flatmap=None):
        from mcmodels.core.cortical_map import CorticalMap
        voxel_array, source_mask, target_mask = cache.get_voxel_connectivity_array()
        vol, _ = cache.get_annotation_volume()
        if custom_flatmap is None:
            M = CorticalMap(projection='dorsal_flatmap')
        else:
            M = NrrdFlatmap(custom_flatmap)
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

    # noinspection PyUnusedLocal
    def cols_and_paths(self, mask2d, bary, shuffle=False):
        nz2d = numpy.nonzero(mask2d)
        cols = bary.col(nz2d[0], nz2d[1])
        paths = self._mapper.paths[self._mapper.view_lookup[mask2d]]
        return cols, paths, zip(*nz2d)

    def paths_in_source_volume(self, mask3d):
        nz3d = numpy.nonzero(mask3d.flat)[0]  # use three_d_to_flat_idx
        return nz3d[numpy.in1d(nz3d, self._source_flat)]  # intersect1d?

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
        pbar = progressbar.ProgressBar(maxval=len(value))
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
            out_R /= out_count
            out_G /= out_count
            out_B /= out_count
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
    def from_cache(cls, cache, custom_flatmap=None):
        from mcmodels.core.cortical_map import CorticalMap
        vol, _ = cache.get_annotation_volume()
        if custom_flatmap is None:
            M = CorticalMap(projection='dorsal_flatmap')
        else:
            M = NrrdFlatmap(custom_flatmap)
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
            out_cols[exp_locs[:, 1] < self._mask2d.shape[1] / 2, :] = numpy.NaN
        else:
            out_cols[exp_locs[:, 1] > self._mask2d.shape[1] / 2, :] = numpy.NaN
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


'''class CustomSourceFlatMapper(VoxelArrayBaryMapper):

    def __init__(self, source_flatmap, voxel_array, source_coords, target_coords,
                 annotation_volume, mapper, structure_tree):
        super(CustomSourceFlatMapper, self).__init__(voxel_array, source_coords, target_coords,
                                                           annotation_volume, mapper, structure_tree)
        self._src_fm = source_flatmap

    @classmethod
    def from_cache(cls, cache, source_flatmap):
        from mcmodels.core.cortical_map import CorticalMap
        voxel_array, source_mask, target_mask = cache.get_voxel_connectivity_array()
        vol, _ = cache.get_annotation_volume()
        M = CorticalMap(projection='dorsal_flatmap')
        tree = cache.get_structure_tree()
        return cls(source_flatmap, voxel_array, source_mask.coordinates, target_mask.coordinates,
                   vol, M, tree)

    def prepare_for_source(self, src, interactive=True, contract=0.75):
        if src == self._prepared_for:
            return
        else:
            self._used_hemisphere = 2
            self._res_cache = {}
            self._prepared_for = src
            self._mask3d = self.make_volume_mask(src)
            nz3d = numpy.nonzero(self._mask3d)
            self._nz2d = self._src_fm[nz3d]
            self._nz3d = numpy.vstack(nz3d).transpose()
            # self.mask_hemisphere(self._nz2d) # TODO: For now masking is done by setting the left hemisphere to NaN in the flatmap volume
            self._nz3d = self._nz3d[~numpy.any(numpy.isnan(self._nz2d), axis=1)]
            self._nz2d = self._nz2d[~numpy.any(numpy.isnan(self._nz2d), axis=1)]
            self._bary = IrregularGridMapper(self._nz2d,
                                             interactive=interactive,
                                             contract=contract)
            self._relevant_paths = self.paths_in_source_volume(self._mask3d)

    def _for_full_volume(self, thresh=1.5, shuffle=False, normalize=True):
        import progressbar
        self._cols = self._bary.col(self._nz2d[:, 0], self._nz2d[:, 1])
        valid = numpy.in1d(self._relevant_paths, self._three_d2flat_idx(self._nz3d))
        self._cols = self._cols[valid]
        self._paths = self._three_d2flat_idx(self._nz3d)[valid]
        value = numpy.abs(numpy.diff(numpy.hstack([self._cols, self._cols[:, 0:1]]), axis=1)).sum(axis=1)
        out_R = numpy.zeros(self._mask3d.shape, dtype=float)
        out_G = numpy.zeros(self._mask3d.shape, dtype=float)
        out_B = numpy.zeros(self._mask3d.shape, dtype=float)
        out_count = numpy.zeros(self._mask3d.shape, dtype=float)
        pbar = progressbar.ProgressBar()
        self._sampled = []

        for _col, _path, _val, _pt in pbar(zip(self._cols, self._paths, value, self._nz2d)):
            if _val > thresh:
                proj = self.proj_for_voxel_3d(_path)
                out_count += proj
                if shuffle:
                    _col = numpy.random.permutation(_col)
                out_R += _col[0] * proj
                out_G += _col[1] * proj
                out_B += _col[2] * proj
                self._sampled.append((_pt[0], _pt[1], _col))

        if normalize:
            out_R /= out_count
            out_G /= out_count
            out_B /= out_count
        return out_R, out_G, out_B'''

