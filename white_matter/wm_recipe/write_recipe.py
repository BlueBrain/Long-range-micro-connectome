from layer_profiles import LayerProfiles, ProfileMixer
from p_types import TreeInnervationModel
from projection_mapping import ProjectionMapper
from projection_strength import ProjectionStrength
from writers import *
from projection_namer import ProjectionNamer
from region_mapper import RegionMapper

fn = 'white_matter_FULL_RECIPE_v0p9.yaml'
l_prof = LayerProfiles()
P = ProjectionStrength()
mixer = ProfileMixer(P)
p_type_mdl = TreeInnervationModel.from_config_file()
mapper = ProjectionMapper()
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

