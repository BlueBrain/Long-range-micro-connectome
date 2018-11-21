#!/usr/bin/env python
import numpy
from white_matter import validate


def conditional_conversion(lst):
    out = []
    for s in lst:
        if numpy.all([_x.isdigit() for _x in s]):
            out.append(int(s))
        else:
            out.append(s)
    return out


def main(fn_feather, fn_circ, show_x='x', show_y='y', show_region='source',
         **kwargs):
    if isinstance(show_x, list):
        show_x = show_x[0]
    if isinstance(show_y, list):
        show_y = show_y[0]
    if isinstance(show_region, list):
        show_region = show_region[0]
    P = validate.ProjectionizerResult(fn_feather, fn_circ)
    nbins = 100
    props = P._circ.v2.cells.get(group=P._pre_gids, properties=kwargs.keys())
    valid = numpy.all(numpy.vstack([numpy.in1d(props[k], v)
                                    for k, v in kwargs.items()]), axis=0)
    pre_gids = props.index.values[valid]
    post_g_xyz = P._postsynaptic_property(pre_gids, ['sgid', 'x', 'y', 'z'])
    pre_s_xyz = P._circ.v2.cells.get(pre_gids, properties=['x', 'y', 'z'])
    pre_xyz = pre_s_xyz.loc[post_g_xyz['sgid']]

    '''
    D = validate.DorsalFlatmap()
    post_proj = D.transform_points(*[post_g_xyz[_s].values for _s in ['x','y','z']])
    pre_proj = D.transform_points(*[pre_xyz[_s].values for _s in ['x','y','z']])'''
    def make_img(data, dx, dy):
        return numpy.array([[numpy.nanmean(data[(dx == i) & (dy == j)])
                             for i in range(1, nbins+2)]
                            for j in range(1, nbins+2)])
    def make_digitize(x, y):
        bx = numpy.linspace(numpy.min(x), numpy.max(x) + 1.0, nbins + 1)
        by = numpy.linspace(numpy.min(y), numpy.max(y) + 1.0, nbins + 1)
        return bx, by, numpy.digitize(x, bins=bx), numpy.digitize(y, bins=by)

    if show_region == 'source':
        bx, by, dx, dy = make_digitize(pre_xyz[show_x].values, pre_xyz[show_y].values)
        IMG = [make_img(post_g_xyz[_s].values, dx, dy) for _s in ['x', 'y', 'z']]
    elif show_region == 'target':
        bx, by, dx, dy = make_digitize(post_g_xyz[show_x].values, post_g_xyz[show_y].values)
        IMG = [make_img(pre_xyz[_s].values, dx, dy) for _s in ['x', 'y', 'z']]
    else:
        raise Exception("Invalid argument for 'show_region': %s.\nValid values:\n\tsource, target" % show_region)

    from matplotlib import pyplot as plt
    for _axis, _I in zip(['x', 'y', 'z'], IMG):
        fig = plt.figure()
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
        img = ax.imshow(_I, extent=[bx[0], bx[-1], by[0], by[-1]])
        plt.colorbar(img).set_label(_axis)
        plt.axis('off')
        ax.plot([bx[0], bx[0] + 100.0], [by[0], by[0]], color='black', lw=5)
        ax.text(bx[0] + 50.0, by[0] - 10.0, show_x, horizontalalignment='center', verticalalignment='top')
        ax.plot([bx[0], bx[0]], [by[0], by[0] + 100], color='black', lw=5)
        ax.text(bx[0]-10, by[0] +50, show_y, color='black', horizontalalignment='right')
        ax.set_title("%s region" % show_region)
    plt.show()

if __name__ == "__main__":
    import sys, os
    if len(sys.argv) < 3:
        print """Usage: %s feather_file CircuitConfig show_x='axis_name', show_y='axis_name',
        fltr_type1=fltr_value1 fltr_type2=fltr_value2, ...
        show_x and show_y determine which axis of the source region is depicted along the x and y axes of the image.
        Filters are neuron property filters (e.g. region=VISp4) that are applied to PREsynaptic neurons""" % \
              os.path.split(__file__)[1]
        sys.exit(2)
    fltr_args = {}
    for arg in sys.argv[3:]:
        splt_arg = arg.split('=')
        fltr_args[splt_arg[0]] = conditional_conversion(splt_arg[1].split(','))
    show_x = fltr_args.pop('show_x', 'x')
    show_y = fltr_args.pop('show_y', 'y')
    show_region = fltr_args.pop('show_region', 'source')
    main(sys.argv[1], sys.argv[2], show_y=show_y, show_x=show_x,
         show_region=show_region, **fltr_args)
