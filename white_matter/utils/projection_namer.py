class ProjectionNamer(object):
    def __init__(self):
        pass

    def projection(self, src_region, src_type, tgt_region, hemi=None):
        return self.half_projection(src_region, src_type, tgt_region) + '_' + hemi

    @staticmethod
    def half_projection(src_region, src_type, tgt_region):
        return src_region + '_' + src_type + '_to_' + tgt_region

    @staticmethod
    def comb_pop(reg_name, src_type):
        return reg_name + '_' + src_type

    @staticmethod
    def comb_hemi(region_name, hemi):
        return region_name + '_' + hemi
