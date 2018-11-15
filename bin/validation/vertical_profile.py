#!/usr/bin/env python
from white_matter import validate
from matplotlib import pyplot as plt
import numpy


def conditional_conversion(lst):
    out = []
    for s in lst:
        if numpy.all([_x.isdigit() for _x in s]):
            out.append(int(s))
        else:
            out.append(s)
    return out


def main(fn_feather, fn_circ, n_smpl, **kwargs):
    D = validate.DorsalFlatmap()
    A = validate.ProjectionizerResult(fn_feather, fn_circ)
    fltrs = dict([(k, v) if isinstance(v, list) else (k, [v])
                  for k, v in kwargs.items()])
    print fltrs
    if len(fltrs) > 0:
        props = A._circ.v2.cells.get(group=A._pre_gids, properties=fltrs.keys())
        valid = numpy.all(numpy.vstack([numpy.in1d(props[k], v)
                                        for k, v in fltrs.items()]), axis=0)
        gid_pre = numpy.random.choice(A._pre_gids[valid],
                                       numpy.minimum(n_smpl, valid.sum()),
                                       replace=False)
    else:
        gid_pre = numpy.random.choice(A._pre_gids,
                                       numpy.minimum(n_smpl, len(A._pre_gids)),
                                       replace=False)

    syn_loc = A.postsynaptic_locations(gid_pre, split=False, unique=False)
    Y = D.transform_points_to_depth(syn_loc['x'].values, syn_loc['y'].values,
                                    syn_loc['z'].values, only_unique=False)
    bins = numpy.linspace(0, 2500, 26)
    H = numpy.histogram(100 * Y[~numpy.isnan(Y)], bins=bins)[0]
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot(H, bins[:-1])
    ax.set_ylabel('Depth (um)')
    ax.set_xlabel('Synapses')
    ax.set_ylim(bins[[-2, 0]])
    plt.show()


if __name__ == "__main__":
    import sys
    import os
    if len(sys.argv) < 3:
        print """Usage: %s feather_file CircuitConfig fltr_type1=fltr_value1 fltr_type2=fltr_value2, ...
        Filters are neuron property filters (e.g. region=VISp4) that are applied to PREsynaptic neurons""" %\
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
