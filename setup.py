from setuptools import setup, find_packages


def find_scripts():
    scripts = [
        "bin/write_wm_recipe.py",
        "bin/write_projection_mapping_cache.py",
        "bin/write_ptype_tree_model_cache.py",
        "bin/write_projection_strength_cache.py",
    ]
    return scripts


setup(
    name="white_matter",
    version="1.40",
    install_requires=[
        "h5py",
        "allensdk@git+https://github.com/AllenInstitute/AllenSDK.git",
        "simplejson",
        "numpy",
        "progressbar",
        "PyYAML",
        "scipy>=1.0.0",
        "networkx",
        "matplotlib",
        "python-louvain",
        "voxcell",
        "mouse-connectivity-models@git+https://github.com/AllenInstitute/mouse_connectivity_models.git"
    ],
    dependency_links = [
      "git+github://github.com/AllenInstitute/mouse_connectivity_models.git",
      "git+https://github.com/AllenInstitute/AllenSDK.git"
    ],
    packages=find_packages(),
    include_package_data=True,
    scripts=find_scripts(),
    author="Blue Brain Project, EPFL",
    description="""Analyze whole-neocortex connectivity in the mouse brain and write a generative recipe""",
    license="BSD-3-Clause",
    keywords=("neuroscience", "brain", "white matter", "yaml", "modelling"),
    url="http://bluebrain.epfl.ch",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "License :: BSD-3-Clause",
        "Operating System :: POSIX",
        "Topic :: Utilities",
    ],
)
