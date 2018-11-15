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
        props = A._circ.v2.cells.get(group=A._post_gids, properties=fltrs.keys())
        valid = numpy.all(numpy.vstack([numpy.in1d(props[k], v)
                                        for k, v in fltrs.items()]), axis=0)
        gid_post = numpy.random.choice(A._post_gids[valid],
                                       numpy.minimum(n_smpl, valid.sum()),
                                       replace=False)
    else:
        gid_post = numpy.random.choice(A._post_gids,
                                       numpy.minimum(n_smpl, len(A._post_gids)),
                                       replace=False)

    syns_con = A.presynaptic_syns_con(gid_post, split=False, lookup_by='brain region')

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    for k, v in syns_con.items():
        ax.plot(range(1, len(v) + 1), v, label=k)
    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel('Synapses per connection')
    ax.set_ylabel('Connections')
    plt.show()


if __name__ == "__main__":
    import sys, os
    if len(sys.argv) < 3:
        print """Usage: %s feather_file CircuitConfig fltr_type1=fltr_value1 fltr_type2=fltr_value2, ...
        Filters are neuron property filters (e.g. region=VISp4) that are applied to POSTsynaptic neurons""" % \
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
