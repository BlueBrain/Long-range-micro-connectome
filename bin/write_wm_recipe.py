#!/usr/bin/env python
from white_matter.wm_recipe.layer_profiles import LayerProfiles, ProfileMixer
from white_matter.wm_recipe.p_types import TreeInnervationModel
from white_matter.wm_recipe.projection_mapping import ProjectionMapper
from white_matter.wm_recipe.projection_strength import ProjectionStrength
from white_matter.wm_recipe.writers import *
from white_matter.wm_recipe.projection_namer import ProjectionNamer
from white_matter.wm_recipe.region_mapper import RegionMapper
from white_matter import __version__


if __name__ == "__main__":
    import sys, os
    if len(sys.argv) > 1:
        cfg_file = sys.argv[1]
        assert os.path.exists(cfg_file)
    else:
        cfg_file = None

    fn = 'white_matter_FULL_RECIPE_v%s.yaml' % __version__.replace('.', 'p')
    l_prof = LayerProfiles(cfg_file=cfg_file)
    P = ProjectionStrength(cfg_file=cfg_file)
    mixer = ProfileMixer(P, cfg_file=cfg_file)
    p_type_mdl = TreeInnervationModel.from_config_file(cfg_file=cfg_file)
    mapper = ProjectionMapper(cfg_file=cfg_file)
    namer = ProjectionNamer()
    mpr = RegionMapper()


    class tab_replacer(object):
        def __init__(self, fid):
            self.fid = fid

        def write(self, txt):
            self.fid.write(txt.replace('\t', '    '))


    with open(fn, 'w') as fid:
        rid = tab_replacer(fid)
        PopulationWriter(mpr, namer)(rid)
        ProjectionWriter(mpr, namer, P, mixer, mapper)(rid)
        PTypeWriter(mpr, namer, p_type_mdl, P)(rid)
        LayerProfileWriter(l_prof)(rid)
        SynapseTypeWriter()(rid)
        ConnectionMappingWriter()(rid)

