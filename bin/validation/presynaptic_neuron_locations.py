#!/usr/bin/env python
from white_matter import validate
from matplotlib import pyplot as plt
import numpy


class PresynNeuronFinder(object):
    def __init__(self, circ):
        import os, h5py
        self._str_n2r = 'edges/default/indices/target_to_source/node_id_to_ranges'
        self._str_r2i = 'edges/default/indices/target_to_source/range_to_edge_id'
        self._str_e2g = 'edges/default/source_node_id'
        self.circ = circ
        self.h5 = h5py.File(os.path.join(circ.config.Run.nrnPath, 'edges.sonata'), 'r')

    def presyn_gids(self, gid):
        if hasattr(gid, '__iter__'):
            return self._presyn_gids(gid)
        R = self.h5[self._str_n2r][gid - 1]
        RR = self.h5[self._str_r2i][R[0]:R[1]]
        out_gids = numpy.hstack([numpy.unique(self.h5[self._str_e2g][_R[0]:_R[1]])
                                 for _R in RR])
        return numpy.unique(out_gids) + 1

    def _presyn_gids(self, gids):
        import progressbar
        pbar = progressbar.ProgressBar()
        out_gids = []
        for gid in pbar(gids):
            out_gids.extend(self.presyn_gids(gid))
        out_gids = numpy.array(out_gids)
        out_gids.sort()
        return out_gids

    def presyn_locations(self, gids):
        p_gids = self.presyn_gids(gids)
        return self.circ.v2.cells.get(group=p_gids, properties=['x', 'y', 'z'])


def conditional_conversion(lst):
    out = []
    for s in lst:
        if numpy.all([_x.isdigit() for _x in s]):
            out.append(int(s))
        else:
            out.append(s)
    return out


def __make_bins__(xlim, ylim, nbins):
    xlim = numpy.sort(xlim); ylim = numpy.sort(ylim)
    xbins = numpy.linspace(xlim[0], xlim[1], nbins[0] + 1)
    ybins = numpy.linspace(ylim[0], ylim[1], nbins[1] + 1)
    return xbins, ybins


def dot_histogram(ax, locs, xlim, ylim, nbins, max_v=5.0):
    cols = plt.cm.Green_r
    xbins, ybins = __make_bins__(xlim, ylim, nbins)
    H = numpy.histogram2d(locs[:, 1], locs[:, 0],
                          (xbins, ybins))[0]
    xc = 0.5 * (xbins[:-1] + xbins[1:]); yc = 0.5 * (ybins[:-1] + ybins[1:])
    nz = numpy.nonzero(H)
    for ix, iy in zip(*nz):
        c = cols[H[ix, iy] / max_v]
        ax.plot(xc[ix], yc[iy], ls='none', marker='h', color=c)


def pick_central_gids(circ, base_gids, N):
    loc = circ.v2.cells.get(group=base_gids, properties=['x', 'y', 'z'])
    mn_loc = loc.values.mean(axis=0)
    sqD = numpy.sum((loc.values - mn_loc) ** 2, axis=1)
    gids = base_gids[numpy.argsort(sqD)[:N]]
    out_loc = loc.loc[gids]
    return gids, out_loc


def main(fn_feather, fn_circ, n_smpl, pick='center',
         include_local=True, **kwargs):
    D = validate.DorsalFlatmap()
    A = validate.ProjectionizerResult(fn_feather, fn_circ)
    fltrs = dict([(k, v) if isinstance(v, list) else (k, [v])
                  for k, v in kwargs.items()])
    print fltrs
    if pick == 'center':
        gid_post, loc_post = pick_central_gids(A._circ, A._post_gids, n_smpl)
    elif pick == 'random':
        gid_post = numpy.random.choice(A._post_gids,
                                       numpy.minimum(n_smpl, len(A._post_gids)),
                                       replace=False)
        loc_post = A._circ.v2.cells.get(group=gid_post, properties=['x', 'y', 'z'])
    else:
        raise Exception("Unknown pick function: %s" % pick)
    loc_pre = A.presynaptic_neuron_locations(gid_post, split=False, unique_neurons=True)
    if len(fltrs) > 0:
        props = A._presynaptic_circ_property(gid_post, fltrs.keys(), unique_neurons=True)
        valid = numpy.all(numpy.vstack([numpy.in1d(props[k], v)
                                        for k, v in fltrs.items()]), axis=0)
        loc_pre = loc_pre.loc[valid]
    if include_local:
        pre_finder = PresynNeuronFinder(A._circ)
        loc_pre.append(pre_finder.presyn_locations(gid_post))

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xticks([]); ax.set_yticks([])
    #[D.draw_region(ax, _reg) for _reg in D._mpr.region_names]
    D.draw_modules(ax, pre_rendered=True)
    proj_post = D.transform_points(loc_post['x'].values, loc_post['y'].values,
                                   loc_post['z'].values)
    proj_pre = D.transform_points(loc_pre['x'].values, loc_pre['y'].values,
                                  loc_pre['z'].values)
    proj_pre += numpy.random.rand(proj_pre.shape[0], proj_pre.shape[1]) - 0.5
    ax.plot(proj_post[:, 1], proj_post[:, 0], 'ob')
    dot_histogram(ax, proj_pre, ax.get_xlim(), ax.get_ylim(), (100, 50))
    #ax.plot(proj_pre[:, 1], proj_pre[:, 0], '.', color=[0.2, 0.75, 0.2])
    plt.show()


if __name__ == "__main__":
    import sys, os
    if len(sys.argv) < 3:
        print """Usage: %s feather_file CircuitConfig fltr_type1=fltr_value1 fltr_type2=fltr_value2, ...
        Filters are neuron property filters (e.g. region=VISp4) that are applied to PREsynaptic neurons""" % \
              os.path.split(__file__)[1]
        sys.exit(2)
    n_smpl = 25
    fltr_args = {}
    for arg in sys.argv[3:]:
        if arg.startswith("n_smpl="):
            n_smpl = int(arg[7:])
        else:
            splt_arg = arg.split('=')
            fltr_args[splt_arg[0]] = conditional_conversion(splt_arg[1].split(','))
    main(sys.argv[1], sys.argv[2], n_smpl, **fltr_args)
