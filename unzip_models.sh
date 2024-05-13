#!/bin/bash

# Unzip the downloaded file into the outputs directory
echo "Extracting top_models.zip into outputs/..."
unzip -q "$OUTPUT_ZIP" -d outputs/

# Move all files within the top_models/ subdirectory to outputs/
mv outputs/top_models/* outputs/

# Remove the now empty top_models/ directory
rmdir outputs/top_models

# Optionally, you can remove the zip file after extraction
# rm "$OUTPUT_ZIP"
