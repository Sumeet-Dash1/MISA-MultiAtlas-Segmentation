# Define paths
label_mni="/Users/sumeetdash/MAIA/Semester_3/CODES/MISA/Lab2_2/MNITemplateAtlas/WM_probability_map.nii.gz"
output_dir="/Users/sumeetdash/MAIA/Semester_3/CODES/MISA/Lab2_2/Registered_MNI_labels/WM"
transform_param_dir="/Users/sumeetdash/MAIA/Semester_3/CODES/MISA/Lab2_2/Registered_MNI_test"

# # Create the output directory if it doesn't exist
# mkdir -p "$output_dir"

# Loop through each label in the training labels directory
for label_file in "$transform_param_dir"/*; do
    # Extract the base name (e.g., 1001_3C from 1001_3C.nii.gz)
    # base_name=$(basename "$label_file" .nii.gz)
    
    # # Remove "_3C" if it exists in the base name
    # base_name="${base_name/_3C/}"

    # Define the corresponding TransformParameters file path
    transform_param_file="$label_file/TransformParameters.1.txt"
    
    # Check if the TransformParameters file exists
    if [[ -f "$transform_param_file" ]]; then
        # Define the output directory for transformed labels
        specific_output_dir="$output_dir/$label_file"
        
        # Create the specific output directory
        mkdir -p "$specific_output_dir"

        # Run transformix with the current label file
        /Users/sumeetdash/MAIA/Semester_3/CODES/MISA/Lab2_2/elastix/transformix \
            -in "$label_mni" \
            -out "$specific_output_dir" \
            -tp "$transform_param_file"

        echo "Transformed $label_file, results stored in $specific_output_dir"
    else
        echo "Warning: TransformParameters file not found for $transform_param_file, skipping..."
    fi
done
