class PopulationWriter(object):
    def __init__(self, mpr, namer):
        self.mpr = mpr
        self.namer = namer

    def write_base_populations(self, fid):
        def single_population(reg_name):
            fid.write('\t- name: ' + self.namer.comb_pop(reg_name, 'ALL_LAYERS') + '\n')
            fid.write('\t  atlas_region:\n\t\t  name: ' + reg_name + '\n')
            fid.write('\t\t  subregions: [l1, l23, l4, l5, l6a, l6b]\n')
            fid.write('\t  filters: []\n\n')
        fid.write('populations:\n')
        for i, nm in enumerate(self.mpr.region_names):
            single_population(nm)
        fid.write('\n')

    def __call__(self, fid):
        def single_population(reg_name, source_name):
            fid.write('\t- name: ' + self.namer.comb_pop(reg_name, source_name) + '\n')
            fid.write('\t  atlas_region:\n\t\t  name: ' + reg_name + '\n')
            fid.write('\t\t  subregions: ' + str(self.mpr.source_layers[source_name]) + '\n')
            fid.write('\t  filters: ' + str(self.mpr.source_filters[source_name]) + '\n\n')

        self.write_base_populations(fid)
        for i, nm in enumerate(self.mpr.region_names):
            for j, src in enumerate(self.mpr.source_names):
                single_population(nm, src)
        fid.write('\n')
