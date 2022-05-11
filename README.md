# Long-range-micro-connectome

This repository deals with generating a whole-neocortex, neuron-to-neuron connectome in a morphologically detailed model of mouse neocortex.

Detailed information can be found in the [documentation](DOCUMENTATION.pdf).

# Installation

To install this repository, first a number of code dependencies have to be installed. The code depends on the python packages _allensdk_ and _mouse-connectivity-models_ of the Allen Institute for Brain Science (AIBS). They can be acquired from their github pages at https://github.com/AllenInstitute . For more detailed instructions and documentation of the packages refer to https://allensdk.readthedocs.io/en/latest/install.html .

Once the manual dependencies have been fulfilled, you can install this repository using the following command

```
pip install git+https://github.com/BlueBrain/Long-range-micro-connectome.git
```

# Examples
As the purpose of this repository is to write a formalization of anatomical constraints on long-range connectivity, it really only has two use cases: Write the constraints as used by Blue Brain in their manuscript, or customize them beforehand.
### Case 1: FORMALIZE ANATOMICAL CONSTRAINTS AS IN THE PUBLISHED MANUSCRIPT
Write a yaml file specifying the anatomical constraints on long-range connectivity in the mouse isocortex.
This will create the file specifying the constraints as white_matter_FULL_RECIPE_v1p20.yaml.
Constraints will be identical to the ones used for the manuscript.
```
write_wm_recipe.py ./configurations/wm-refined-neocortex_template.json
```

### Case 2: FORMALIZE ANATOMICAL CONSTRAINTS WITH CUSTOM CHANGES
Write a yaml file specifying the anatomical constraints, but based on different anatomical data, assumptions or parcellations.
```
CUSTOM_FILE=/path/to/my/custom_template.json
cp ./configurations/wm-refined-neocortex_template.json $CUSTOM_FILE
```
At this point edit CUSTOM_FILE to update the underlying anatomical data to your liking. Refer to the [documentation](DOCUMENTATION.pdf) for details on the json tempplate file format. Then:
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

Copyright (c) 2022-2022 Blue Brain Project/EPFL

