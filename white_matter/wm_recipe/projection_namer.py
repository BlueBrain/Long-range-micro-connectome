class ProjectionNamer(object):
    def __init__(self):
        pass

    def projection(self, src_region, src_type, tgt_region, hemi=None):
         return self.half_projection(src_region, src_type, tgt_region) + '_' + hemi

    def half_projection(self, src_region, src_type, tgt_region):
        return src_region + '_' + src_type + '_to_' + tgt_region

    def comb_pop(self, reg_name, src_type):
        return reg_name + '_' + src_type

    def comb_hemi(self, region_name, hemi):
        return region_name + '_' + hemi
