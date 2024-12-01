# Set paths and parameter files
fixed_image_dir="../../Data/Validation_Set/"
output_dir="../data/processed"
param_affine="./Par0009/Parameters.Par0009.affine.txt"
param_elastic="/Users/sumeetdash/MAIA/Semester_3/CODES/MISA/Lab2_2/parameters/models/Par0009/Parameters.Par0009.elastic.txt"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through each image in the training set
for moving_image in ../../Data/Training_Set/testing-images/*.nii.gz; do
    # Skip the fixed image if it's also in the moving images
    if [[ "$moving_image" == "$fixed_image" ]]; then
        continue
    fi

    # Extract the base name of the moving image (e.g., 1001 from 1001.nii.gz)
    base_name=$(basename "$fixed_image" .nii.gz)

    # Define the specific output folder for this registration
    specific_output_dir="$output_dir/$base_name"
    
    # Create the specific output directory
    mkdir -p "$specific_output_dir"

    # Run elastix with the current moving image
    ./elastix/elastix \
        -f "$fixed_image" \
        -m "$moving_image" \
        -out "$specific_output_dir" \
        -p "$param_affine" \
        -p "$param_elastic"

    echo "Processed $moving_image, results stored in $specific_output_dir"
done
