# Set paths and parameter files
fixed_image_dir="../Data/SPM/Input/Validation_Set/"
moving_image_dir="../Data/SPM/Input/Training_Set/"
output_dir="../Data/SPM/processed_update/"
param_affine="./Par0009/Parameters.Par0009.affine.txt"
param_elastic="./Par0009/Parameters.custom.elastic.txt"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through each fixed image (validation set)
for fixed_folder in "$fixed_image_dir"IBSR_*; do
    # Define the fixed image path
    fixed_image="$fixed_folder/m$(basename "$fixed_folder").nii"

    # Check if the fixed image file exists
    if [[ ! -f "$fixed_image" ]]; then
        echo "Fixed image not found: $fixed_image. Skipping..."
        continue
    fi

    # Loop through each moving image (training set)
    for moving_folder in "$moving_image_dir"IBSR_*; do
        # Define the moving image path
        moving_image="$moving_folder/m$(basename "$moving_folder").nii"

        # Check if the moving image file exists
        if [[ ! -f "$moving_image" ]]; then
            echo "Moving image not found: $moving_image. Skipping..."
            continue
        fi

        # Skip if fixed image and moving image are the same
        fixed_base=$(basename "$fixed_folder")
        moving_base=$(basename "$moving_folder")
        if [[ "$fixed_base" == "$moving_base" ]]; then
            continue
        fi

        # Define the specific output folder for this registration
        specific_output_dir="$output_dir/${fixed_base}/${moving_base}_to_${fixed_base}/"

        # Create the specific output directory
        mkdir -p "$specific_output_dir"

        # Run elastix with the current fixed and moving image
        ../elastix/elastix \
            -f "$fixed_image" \
            -m "$moving_image" \
            -out "$specific_output_dir" \
            -p "$param_affine" \
            -p "$param_elastic"

        echo "Processed $moving_base (moving) with $fixed_base (fixed), results stored in $specific_output_dir"
    done
done