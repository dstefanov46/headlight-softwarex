#!/bin/bash

set -e  # Exit on any error
set -o pipefail

# Get required arguments
FOLDER_PATH="$1"
DOCKER_USER="$2"
IMAGE_NAME="$3"
IMAGE_TAG="$4"

# Validate input
if [[ -z "$FOLDER_PATH" ]]; then
    echo "ERROR: Folder path not provided." >&2
    exit 1
fi
if [[ -z "$DOCKER_USER" ]]; then
    echo "ERROR: Docker Hub username not provided." >&2
    exit 1
fi
if [[ -z "$IMAGE_NAME" ]]; then
    echo "ERROR: Image name not provided." >&2
    exit 1
fi
if [[ -z "$IMAGE_TAG" ]]; then
    echo "ERROR: Image tag not provided." >&2
    exit 1
fi

# Prepare list of copied files for cleanup
COPIED_FILES=()

# Handle extra files from $5 onward
shift 4  # Shift to the extra files
for EXTRA_FILE in "$@"; do
    if [[ ! -f "$EXTRA_FILE" ]]; then
        echo "ERROR: Extra file not found at $EXTRA_FILE" >&2
        exit 1
    fi
    FILENAME=$(basename "$EXTRA_FILE")
    cp "$EXTRA_FILE" "$FOLDER_PATH/$FILENAME"
    if [[ $? -ne 0 ]]; then
        echo "ERROR: Failed to copy $EXTRA_FILE to $FOLDER_PATH" >&2
        exit 1
    fi
    COPIED_FILES+=("$FOLDER_PATH/$FILENAME")
done

# Change into the folder
cd "$FOLDER_PATH" || { echo "ERROR: Cannot change directory to $FOLDER_PATH" >&2; exit 1; }

# Build the Docker image
echo "Building Docker image $DOCKER_USER/$IMAGE_NAME:$IMAGE_TAG ..."
if ! docker build -t "$DOCKER_USER/$IMAGE_NAME:$IMAGE_TAG" .; then
    echo "ERROR: Docker build failed." >&2
    exit 1
fi

# Push the Docker image
echo "Pushing Docker image to Docker Hub..."
if ! docker push "$DOCKER_USER/$IMAGE_NAME:$IMAGE_TAG"; then
    echo "ERROR: Docker push failed." >&2
    exit 1
fi

echo "DONE: Docker image pushed successfully."

# Cleanup copied files
for FILE in "${COPIED_FILES[@]}"; do
    rm -f "$FILE"
done
