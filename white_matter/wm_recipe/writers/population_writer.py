class PopulationWriter(object):
    def __init__(self, mpr, namer):
        self.mpr = mpr
        self.namer = namer

    def write_base_populations(self, fid):
        def single_population(reg_name):
            fid.write('\t- name: ' + self.namer.comb_pop(reg_name, 'ALL_LAYERS') + '\n')
            fid.write('\t  atlas_region:\n\t\t  name: ' + reg_name + '\n')
            fid.write('\t\t  subregions: [l1, l2, l3, l4, l5, l6]\n')
            fid.write('\t  filters: []\n\n')
        fid.write('populations:\n')
        for i, nm in enumerate(self.mpr.region_names):
            single_population(nm)
        fid.write('\n')

    def __dict2str__(self, a_dict):
        out_dict = {}
        for k, v in a_dict.items():
            if isinstance(v, dict):
                out_dict[str(k)] = self.__dict2str__(v)
            elif isinstance(v, list):
                out_dict[str(k)] = map(str, v)
            else:
                out_dict[str(k)] = str(v)
        return out_dict

    def __source_filters_str__(self, source_name):
        return str(self.__dict2str__(self.mpr.source_filters[source_name]))

    def __source_layer_str__(self, source_name):
        return str(map(str, self.mpr.source_layers[source_name]))

    def __call__(self, fid):
        def single_population(reg_name, source_name):
            fid.write('\t- name: ' + self.namer.comb_pop(reg_name, source_name) + '\n')
            fid.write('\t  atlas_region:\n\t\t  name: ' + reg_name + '\n')
            fid.write('\t\t  subregions: ' + self.__source_layer_str__(source_name) + '\n')
            fid.write('\t  filters: ' + self.__source_filters_str__(source_name) + '\n\n')

        self.write_base_populations(fid)
        for i, nm in enumerate(self.mpr.region_names):
            for j, src in enumerate(self.mpr.source_names):
                single_population(nm, src)
        fid.write('\n')
