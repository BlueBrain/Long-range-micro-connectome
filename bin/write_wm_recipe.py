#!/usr/bin/env python
from white_matter.wm_recipe.layer_profiles import LayerProfiles, ProfileMixer
from white_matter.wm_recipe.p_types import TreeInnervationModelCollection
from white_matter.wm_recipe.projection_mapping import ProjectionMapper
from white_matter.wm_recipe.projection_strength import ProjectionStrength
from white_matter.wm_recipe.synapse_types import SynapseTypes
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
    p_type_mdl = TreeInnervationModelCollection.from_config_file(cfg_file=cfg_file)
    mapper = ProjectionMapper(cfg_file=cfg_file)
    syn_types = SynapseTypes(cfg_file=cfg_file)
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
        ProjectionWriter(mpr, namer, P, mixer, mapper, syn_types)(rid)
        PTypeWriter(mpr, namer, p_type_mdl, P, interaction_thresh=2.0)(rid)
        LayerProfileWriter(l_prof)(rid)
        SynapseTypeWriter(syn_types)(rid)
        ConnectionMappingWriter()(rid)

