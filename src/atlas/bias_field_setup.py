import os
import gzip
import shutil

# Source and destination directories
source_dir = "/Users/sumeetdash/MAIA/Semester_3/CODES/MISA/Final_Project/MISA-MultiAtlas-Segmentation/Data/Training_Set"
destination_dir = "/Users/sumeetdash/MAIA/Semester_3/CODES/MISA/Final_Project/MISA-MultiAtlas-Segmentation/Data/SPM/Input/Training_Set"

def decompress_nii_gz(source_path, dest_path):
    """Decompress a .nii.gz file and save it as .nii"""
    with gzip.open(source_path, 'rb') as f_in:
        with open(dest_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Process each folder in the source directory
for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    
    # Skip if not a directory
    if not os.path.isdir(folder_path):
        continue
    
    # Create the corresponding folder in the destination directory
    dest_folder_path = os.path.join(destination_dir, folder_name)
    os.makedirs(dest_folder_path, exist_ok=True)
    
    # Process each .nii.gz file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".nii.gz"):
            source_file_path = os.path.join(folder_path, file_name)
            dest_file_path = os.path.join(dest_folder_path, file_name.replace(".nii.gz", ".nii"))
            
            # Decompress and save the file
            decompress_nii_gz(source_file_path, dest_file_path)

print("Decompression completed successfully.")