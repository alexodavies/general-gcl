#!/bin/bash

MODELS_URL="https://drive.google.com/uc?export=download&id=1Ionm2UsVLNpPmQdOiBBzGq_YjELllGeC"


# Create the outputs directory if it doesn't exist
mkdir -p outputs

# Download the zip file
echo "Downloading top_models.zip..."
wget -O top_models.zip "$MODELS_URL"

# Unzip the downloaded file into the outputs directory
echo "Extracting top_models.zip into outputs/..."
unzip -q top_models.zip -d outputs/

# Move all files within the top_models/ subdirectory to outputs/
mv outputs/top_models/* outputs/

# Remove the now empty top_models/ directory
rmdir outputs/top_models

# Optionally, you can remove the zip file after extraction
rm top_models.zip
