#!/bin/bash
#SBATCH --job-name=MergeSN
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks=1 --cpus-per-task=32
#SBATCH --time=2:00:00
#SBATCH --output=./slurm_logs/merge/shapenet_%A.out
#SBATCH --error=./slurm_logs/merge/shapenet_%A.err

if ! command -v pigz &> /dev/null; then
    echo "Error: pigz is not installed. Install it and rerun the script."
    exit 1
fi

################### Change these! ################### 
BASE_DIR="/cluster/work/riner/users/asgobbi/datasets"
SHARD_DIR="$BASE_DIR/renders/shapenet/shards"
DATASET="sn_224"
#####################################################

TARBALL="$BASE_DIR/$DATASET.tar.gz"
TEMP_DIR=$(mktemp -d -t $DATASET.XXXXXXXX)
WORKERS=32

# Extract each tarball and merge into the temporary directory
echo "Merging tarballs from $SHARD_DIR ..."
for TARFILE in $SHARD_DIR/shard_*.tar; do
    echo "Processing $TARFILE ..."
    tar -xf $TARFILE -C $TEMP_DIR
done

#Compress the merged content into the final tarball
echo "Creating final compressed tarball: $TARBALL"
tar -cf - -C $TEMP_DIR . | pigz -p $WORKERS > $TARBALL

echo "Cleaning up temporary files..."
rm -rf $TEMP_DIR
echo "Done!"