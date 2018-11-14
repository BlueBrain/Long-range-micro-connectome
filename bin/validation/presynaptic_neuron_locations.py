from white_matter import validate
from matplotlib import pyplot as plt
import numpy


def main(fn_feather, fn_circ, n_smpl, **kwargs):
    D = validate.DorsalFlatmap()
    A = validate.ProjectionizerResult(fn_feather, fn_circ)
    fltrs = dict([(k, v) if isinstance(v, list) else (k, [v])
                  for k, v in kwargs.items()])
    print fltrs
    gid_post = numpy.random.choice(A._post_gids,
                                   numpy.minimum(n_smpl, len(A._post_gids)),
                                   replace=False)
    loc_post = A._circ.v2.cells.get(group=gid_post, properties=['x', 'y', 'z'])
    loc_pre = A.presynaptic_neuron_locations(gid_post, split=False, unique_neurons=True)
    if len(fltrs) > 0:
        props = A._presynaptic_circ_property(gid_post, fltrs.keys(), unique_neurons=True)
        valid = numpy.all(numpy.vstack([numpy.in1d(props[k], v)
                                        for k, v in fltrs.items()]), axis=0)
        loc_pre = loc_pre.loc[valid]
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    [D.draw_region(ax, _reg) for _reg in D._mpr.region_names]
    D.draw_modules(ax)
    proj_post = D.transform_points(loc_post['x'].values, loc_post['y'].values,
                                   loc_post['z'].values)
    proj_pre = D.transform_points(loc_pre['x'].values, loc_pre['y'].values,
                                  loc_pre['z'].values)
    proj_pre += numpy.random.rand(proj_pre.shape[0], proj_pre.shape[1]) - 0.5
    ax.plot(proj_post[:, 1], proj_post[:, 0], 'ob')
    ax.plot(proj_pre[:, 1], proj_pre[:, 0], '.', color=[0.2, 0.75, 0.2])
    plt.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print """Usage: %s feather_file CircuitConfig fltr_type1=fltr_value1 fltr_type2=fltr_value2, ...
        Filters are neuron property filters (e.g. region=VISp4) that are applied to PREsynaptic neurons""" % __file__
        sys.exit(2)
    n_smpl = 25
    fltr_args = {}
    for arg in sys.argv[3:]:
        if arg.startswith("n_smpl="):
            n_smpl = int(arg[7:])
        else:
            splt_arg = arg.split('=')
            fltr_args[splt_arg[0]] = splt_arg[1].split(',')
    main(sys.argv[1], sys.argv[2], n_smpl, **fltr_args)
