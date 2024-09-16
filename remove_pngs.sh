#!/bin/bash

# Check if the user provided a directory
if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

# Assign the directory to a variable
TARGET_DIR="$1"

# Find and remove all .png files in the directory and its subdirectories
find "$TARGET_DIR" -type f -name "*.png" -exec rm -v {} \;

# Print completion message
echo "All .png files removed from $TARGET_DIR and its subdirectories."
