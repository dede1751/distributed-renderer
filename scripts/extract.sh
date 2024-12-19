#!/bin/bash

if ! command -v pigz &> /dev/null; then
    echo "Error: pigz is not installed. Install it and rerun the script."
    exit 1
fi

################### Change these! ################### 
BASE_DIR="/cluster/work/riner/users/asgobbi/datasets"
DATASET="sn_224"
#####################################################

TARBALL="$BASE_DIR/$DATASET.tar.gz"
TEMP_DIR=$(mktemp -d -t $DATASET.XXXXXXXX)
WORKERS=32

echo "Extracting tarballs from $TARBALL ..."
pigz -dc -p $WORKERS $TARBALL | tar -xvf - -C $TEMP_DIR
echo "Done!"