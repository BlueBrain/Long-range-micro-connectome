#!/usr/bin/env python
import numpy
from white_matter import validate
from mcmodels.core import VoxelModelCache
from matplotlib import pyplot as plt


projection_root = '/gpfs/bbp.cscs.ch/project/proj68/usr/reimann'


def conditional_conversion(arg):
    if numpy.all([_x.isdigit() for _x in arg]):
        return int(arg)
    else:
        return arg


def main(fn_feather, fn_circ, region_source, region_target, manifest_file=None,
         out_dir=None):
    if manifest_file is None:
        manifest_file = os.path.join(os.getenv('HOME'), 'data/mcmodels/cache.json')
    cache = VoxelModelCache(manifest_file=manifest_file)
    P = validate.ProjectionizerResult(fn_feather, fn_circ)

    proj = validate.ProjectionResultBaryMapper.from_cache(cache, P, projection_root=projection_root)

    proj.prepare_for_source(region_source, interactive=False)
    fig_src = plt.gcf()
    fig_tgt = proj.draw_projection(region_target).figure
    if out_dir is not None:
        fig_src.savefig(os.path.join(out_dir, 'result_proj_%s_%s-source.pdf' %
                                     (region_source, region_target)))
        fig_tgt.savefig(os.path.join(out_dir, 'result_proj_%s_%s-target.pdf' %
                                     (region_source, region_target)))
    if out_dir is None:
        plt.show()

if __name__ == "__main__":
    import sys, os
    if len(sys.argv) < 3:
        print """Usage: %s feather_file CircuitConfig region_source='region_name', region_target='region_name',
        out_dir=None
        region_source and region_target determine which projection is shown.
        Specify out_dir to save the plots in the specified directory.""" % \
              os.path.split(__file__)[1]
        sys.exit(2)
    fltr_args = {}
    out_dir = None
    for arg in sys.argv[3:]:
        if '=' in arg:
            splt_arg = arg.split('=')
            fltr_args[splt_arg[0]] = conditional_conversion(splt_arg[1])
        else:
            raise Exception("Unsupported argument: %s" % arg)
    region_source = fltr_args.pop('region_source', None)
    region_target = fltr_args.pop('region_target', None)
    if region_source is None or region_target is None:
        raise Exception("Need to specify both source and target regions")
    main(sys.argv[1], sys.argv[2], region_source, region_target, **fltr_args)
