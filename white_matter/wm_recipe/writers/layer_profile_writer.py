class LayerProfileWriter(object):
    def __init__(self, layer_profiles):
        self.layer_profiles = layer_profiles

    @staticmethod
    def __layers2str__(layers):
        return '[' + ', '.join(layers) + ']'

    def __call__(self, fid):
        l_profiles = self.layer_profiles
        fid.write('layer_profiles:\n')
        prof_name_pat = 'profile_%d'
        prof_k = sorted(l_profiles.patterns.keys())
        prof_layers = l_profiles.pattern_layers
        for k in prof_k:
            fid.write('\t- name: %s\n' % (prof_name_pat % k))
            fid.write('\t  relative_densities:\n')
            for l, v in zip(prof_layers, l_profiles.patterns[k].transpose()[0]):
                fid.write('\t\t- layers: %s\n' % self.__layers2str__(l))
                fid.write('\t\t  value: %f\n' % v)
            fid.write('\n')

        fid.write('\t- name: %s\n' % (prof_name_pat % l_profiles.fallback_profile["index"]))
        fid.write('\t  relative_densities:\n')
        for entry in l_profiles.fallback_profile["relative_densities"]:
            fid.write('\t\t- layers: %s\n' % self.__layers2str__(entry["layers"]))
            fid.write('\t\t  value: %f\n' % entry["value"])
        fid.write('\n')

        fid.write('\n')
