
import os
import zipfile
import gdown

# Original view URL
dataset_url = "https://drive.google.com/file/d/1wnj8yXOeC1LdeHYZy15P4t4jAK90NgMu/view?usp=drive_link"

# Extract file ID from the URL
file_id = dataset_url.split('/d/')[1].split('/')[0]

# Define the datasets directory
datasets_dir = "datasets"

# Create the datasets directory if it doesn't exist
if not os.path.exists(datasets_dir):
    os.makedirs(datasets_dir)
    print(f"Created directory: {datasets_dir}")
else:
    print(f"Directory {datasets_dir} already exists")

# Download the zip file using gdown
print("Downloading dataset...")
zip_path = os.path.join(datasets_dir, "dataset.zip")
gdown.download(id=file_id, output=zip_path, quiet=False)

print(f"Downloaded to: {zip_path}")

# Unzip the file
print("Unzipping dataset...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(datasets_dir)

print("Dataset unzipped successfully!")

# Optionally, remove the zip file after extraction
os.remove(zip_path)
print("Removed the zip file.")