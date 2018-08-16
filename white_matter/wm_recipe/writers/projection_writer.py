import numpy

class ProjectionWriter(object):
    def __init__(self, mpr, namer, proj_str, profile_mixer, mapper, syn_types):
        self.mpr = mpr
        self.namer = namer
        self.proj_str = proj_str
        self.profile_mixer = profile_mixer
        self.mapper = mapper
        self.syn_types = syn_types

    '''def iterate_per_module(self, reg_fr, source_name, hemi):
        row = self.proj_str(src_type=source_name, hemi=hemi, measurement='connection density')[self.mpr.region2idx(reg_fr)]
        for tgt_module in self.mpr.module_names:
            base_name = self.namer.half_projection(reg_fr, source_name, tgt_module)
            reg_tgts = self.mpr.module2regions(tgt_module)
            val_tgts = row[self.mpr.module2idx(tgt_module)]
            is_valid = numpy.array([_v > 0 and ((_r != reg_fr) or hemi == 'contra')
                                    for _r, _v in zip(reg_tgts, val_tgts)])
            if not numpy.any(is_valid):
                continue
            reg_tgts = [self.namer.comb_pop(_r, 'ALL_LAYERS') for
                        _r, _l in zip(reg_tgts, is_valid) if _l]
            val_tgts = val_tgts[is_valid]
            l_prof = self.profile_mixer.max_module(source_name, self.mpr.region2module(reg_fr), tgt_module) + 1
            yield base_name, reg_tgts, val_tgts, l_prof'''

    def iterate_per_region(self, reg_fr, source_name, hemi):
        row = self.proj_str(src_type=source_name, hemi=hemi, measurement='connection density')[self.mpr.region2idx(reg_fr)]
        for tgt_region in self.mpr.region_names:
            base_name = self.namer.half_projection(reg_fr, source_name, tgt_region)
            val_tgt = row[self.mpr.region2idx(tgt_region)]
            is_valid = (val_tgt > 0) and ((tgt_region != reg_fr) or hemi == 'contra')
            if not is_valid:
                continue
            tgt_full_name = self.namer.comb_pop(tgt_region, 'ALL_LAYERS')
            l_prof = self.profile_mixer.max_module(source_name, self.mpr.region2module(reg_fr),
                                                   self.mpr.region2module(tgt_region)) + 1
            yield base_name, tgt_region, tgt_full_name, val_tgt, l_prof

    def __call__(self, fid):
        str_src_fltr = '\t\t  source_filters: []\n'
        str_proj_name = '\t\t- projection_name: %s\n'
        str_hemi = '\t\t  hemisphere: %s\n'
        str_pops = '\t\t  populations: %s\n'
        str_dens = '\t\t  densities: %s\n'
        str_pop = '\t\t  population: %s\n'
        str_den = '\t\t  densitiy: %s\n'
        str_l_prof = '\t\t  target_layer_profiles:\n\t\t\t- name: %s\n\t\t\t  fraction: 1.0\n'
        str_s_type = '\t\t  synapse_types:\n\t\t\t- name: %s\n\t\t\t  fraction: 1.0\n'

        str_src_coords = '\t\t- base_system: %s\n\t\t  x: %s\n\t\t  y: %s\n'
        str_tgt_coords = '\t\t\t\t- base_system: %s\n\t\t\t\t  x: %s\n\t\t\t\t  y: %s\n'
        str_presyn_mapping = '\t\t  presynaptic_mapping:\n\t\t\t- mapping_variance: %6.5f\n'
        str_con_mapping = '\t\t  connection_mapping:\n\t\t\t- type: type_1\n'

        def single_entry(base_name, hemisphere,
                         tgt, tgt_full_name,
                         density, l_prof,
                         mapping_data, syn_type):
            fid.write(str_proj_name % self.namer.comb_hemi(base_name, hemisphere))
            fid.write(str_src_fltr)
            fid.write(str_hemi % hemisphere)
            fid.write(str_pop % tgt_full_name)
            fid.write(str_den % ("%6.5f" % density))
            x, y, base_sys, m_var = mapping_data(tgt, hemisphere)
            fid.write(str_presyn_mapping % m_var)
            coordinate_system((x, y, base_sys), str_tgt_coords, '\t\t\t  ')
            fid.write(str_con_mapping)
            fid.write(str_l_prof % ("profile_%d" % l_prof))
            fid.write(str_s_type % str(syn_type))
            fid.write('\n')

        def coordinate_system(coord_sys_data, pat, prefix):
            x, y, base_sys = coord_sys_data
            fid.write(prefix + 'mapping_coordinate_system:\n')
            str_x = '[%5.3f, %5.3f, %5.3f]' % tuple(x)
            str_y = '[%5.3f, %5.3f, %5.3f]' % tuple(y)
            fid.write(pat % (base_sys, str_x, str_y))

        func = self.iterate_per_region

        fid.write('projections:\n')
        for reg_fr in self.mpr.region_names:
            mapping_data = self.mapper.for_target(reg_fr)
            for source_name in self.mpr.source_names:
                fid.write('\t- source: ' + self.namer.comb_pop(reg_fr, source_name) + '\n')
                coordinate_system(self.mapper.for_source(reg_fr), str_src_coords, '\t  ')
                fid.write('\t  targets:\n')
                for hemi in ['ipsi', 'contra']:
                    for base_name, tgt, tgt_full_name, dens_value, l_prof in func(reg_fr, source_name, hemi):
                        single_entry(base_name, hemi, tgt, tgt_full_name,
                                     dens_value, l_prof, mapping_data, self.syn_types[source_name])
        fid.write('\n')