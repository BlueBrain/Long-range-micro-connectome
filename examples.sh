# Case 1: FORMALIZE ANATOMICAL CONSTRAINTS AS IN THE PUBLISHED MANUSCRIPT
# Write a yaml file specifying the anatomical constraints on long-range connectivity in the mouse isocortex.
# This will create the file specifying the constraints as white_matter_FULL_RECIPE_v1p20.yaml.
# Constraints will be identical to the ones used for the manuscript.
write_wm_recipe.py ./configurations/default_template.json


# Case 2: FORMALIZE ANATOMICAL CONSTRAINTS WITH CUSTOM CHANGES
# Write a yaml file specifying the anatomical constraints, but based on different anatomical data, assumptions
# or parcellations.
CUSTOM_FILE=/path/to/my/custom_template.json
cp ./configurations/default_template.json $CUSTOM_FILE
# ...Edit CUSTOM_FILE to update the underlying anatomical data. Then:
write_projection_mapping_cache.py $CUSTOM_FILE  # Update cached data related to the spatial mapping of long-range pathways
write_projection_strength_cache.py $CUSTOM_FILE  # Update cached data related to the strength of long-range pathways
write_ptype_tree_model_cache.py $CUSTOM_FILE  # Update cached data related to the common innervation of brain regions by other regions

write_wm_recipe.py $CUSTOM_FILE  # Finally, write the .yaml output file
