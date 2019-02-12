from setuptools import setup, find_packages


def find_scripts():
    scripts = ['bin/write_wm_recipe.py',
               'bin/write_projection_mapping_cache.py',
               'bin/write_ptype_tree_model_cache.py',
               'bin/write_projection_strength_cache.py',
               'bin/proj2csc.py',
               'bin/validation/presynaptic_neuron_locations.py',
               'bin/validation/presynaptic_synapses_per_connection.py',
               'bin/validation/pre_post_mapping.py',
               'bin/validation/vertical_profile.py',
               'bin/validation/pre_post_mapping_cols.py']
    return scripts


setup(
    name='white_matter',
    version='1.15',
    install_requires=['h5py', 'allensdk==0.14.5', 'simplejson', 'mouse-connectivity-models==0.0.1',
                      'numpy', 'progressbar', 'PyYAML', 'scipy==1.0.0', 'networkx', 'matplotlib', 'python-louvain', ],
    packages=find_packages(),
    include_package_data=True,
    scripts=find_scripts(),
    author='Michael Reimann',
    author_email='michael.reimann@epfl.ch',
    description='''Analyze whole-neocortex connectivity in the mouse brain and write a generative recipe''',
    license='Restricted',
    keywords=('neuroscience',
              'brain',
              'white matter',
              'yaml',
              'modelling'),
    url='http://bluebrain.epfl.ch',
    classifiers=['Development Status :: 4 - Beta',
                 'Environment :: Console',
                 'License :: Proprietary',
                 'Operating System :: POSIX',
                 'Topic :: Utilities',
                 ],
)
