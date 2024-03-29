# Long-range-micro-connectome

This repository deals with generating a whole-neocortex, neuron-to-neuron connectome in a morphologically detailed model of mouse neocortex.

Detailed information can be found in the [documentation](DOCUMENTATION.pdf).

# Installation

To install this repository, first a number of code dependencies have to be installed. The code depends on the python packages _allensdk_ and _mouse-connectivity-models_ of the Allen Institute for Brain Science (AIBS). They can be acquired from their github pages at https://github.com/AllenInstitute/AllenSDK and https://github.com/AllenInstitute/mouse_connectivity_models respectively. 

Installing them and their respective dependencies can - to the best of our knowledge - be done with the following commands:
```
pip install six
pip install numpy
pip install sklearn
pip install allensdk@git+https://github.com/AllenInstitute/AllenSDK.git
pip install markupsafe==2.0.1
pip install mouse-connectivity-models@git+https://github.com/AllenInstitute/mouse_connectivity_models.git
```

In case of difficulties with the installation of these dependencies, please refer to the AIBS documentation at https://allensdk.readthedocs.io/en/latest/ .

Once the manual dependencies have been fulfilled, you can install this repository using the following command

```
pip install git+https://github.com/BlueBrain/Long-range-micro-connectome.git
```

# Examples
As the purpose of this repository is to write a formalization of anatomical constraints on long-range connectivity, it really only has two use cases: Write the constraints as used by Blue Brain in their manuscript, or customize them beforehand.
### Case 1: FORMALIZE ANATOMICAL CONSTRAINTS AS IN THE PUBLISHED MANUSCRIPT
Write a yaml file specifying the anatomical constraints on long-range connectivity in the mouse isocortex.
This will create the file specifying the constraints as _white_matter_FULL_RECIPE_v1p20.yaml_.
Constraints will be identical to the ones used for the manuscript.
```
write_wm_recipe.py ./configurations/wm-refined-neocortex_template.json
```
__Note__: In lines 542 and 16069 of "wm-refined-neocortex_template.json" the path to a file called "voxel_model_manifest.json" is specified. This is a file used by the AIBS mouse-connectivity-models to configure the locations of certain cache and data files. If you have previously worked with the AIBS voxel model you can point the entries to your existing manifest file. This will avoid duplicate downloads of data from AIBS servers. If the specified file does not exist it will be created. By default (if you do not modify "wm-refined-neocortex_template.json"), the file will be created in the local directory. However note that a number of data files will also be downloaded to that location, so you might want to point the entry to another location. 

### Case 2: FORMALIZE ANATOMICAL CONSTRAINTS WITH CUSTOM CHANGES
Write a yaml file specifying the anatomical constraints, but based on different anatomical data, assumptions or parcellations.
```
CUSTOM_FILE=/path/to/my/custom_template.json
cp ./configurations/wm-refined-neocortex_template.json $CUSTOM_FILE
```
At this point edit _CUSTOM_FILE_ to update the underlying anatomical data to your liking. Refer to the [documentation](DOCUMENTATION.pdf) for details on the json template file format. Then:
```
# Update cached data related to the spatial mapping of long-range pathways
write_projection_mapping_cache.py $CUSTOM_FILE  

# Update cached data related to the strength of long-range pathways
write_projection_strength_cache.py $CUSTOM_FILE  

# Update cached data related to the common innervation of brain regions by other regions
write_ptype_tree_model_cache.py $CUSTOM_FILE 

# Finally, write the .yaml output file
write_wm_recipe.py $CUSTOM_FILE  
```

# Funding & Acknowledgment

The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

Copyright (c) 2022 Blue Brain Project/EPFL

