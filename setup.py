from setuptools import setup, find_packages


def find_scripts():
    scripts = ['bin/write_wm_recipe.py',
               'bin/write_projection_mapping_cache.py',
               'bin/write_ptype_tree_model_cache.py']
    return scripts


setup(
    name='white_matter',
    version='0.98',
    install_requires=['h5py', 'allensdk==0.14.5', 'simplejson', 'mouse-connectivity-models==0.0.1',
                      'numpy', 'progressbar', 'PyYAML', 'scipy==1.0.0', ],
    packages=find_packages(),
    include_package_data=True,
    scripts=find_scripts(),
    author='Michael Reimann',
    author_email='michael.reimann@epfl.ch',
    description='Analyze white matter connectivity in the mouse brain and write a generative recipe',
    license='Restricted',
    keywords=('neuroscience',
              'brain',
              'white matter',
              'yaml',
              'modelling'),
    url='http://bluebrain.epfl.ch',
    classifiers=['Development Status :: 3 - Alpha',
                 'Environment :: Console',
                 'License :: Proprietary',
                 'Operating System :: POSIX',
                 'Topic :: Utilities',
                 ],
)
