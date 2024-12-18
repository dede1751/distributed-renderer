#!/bin/bash
#SBATCH --job-name=MergeZS
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks=1 --cpus-per-task=32
#SBATCH --time=2:00:00
#SBATCH --output=./slurm_logs/merge/zeroshape_%A.out
#SBATCH --error=./slurm_logs/merge/zeroshape_%A.err

BASE_DIR="/cluster/work/riner/users/asgobbi/datasets"
SHARD_DIR="${BASE_DIR}/renders/zeroshape/shards"
FINAL_TARBALL="${BASE_DIR}/zs_224.tar.gz"
TEMP_DIR=$(mktemp -d)
WORKERS=32

if [ ! -d ${SHARD_DIR} ]; then
    echo "Error: Shard directory does not exist: ${SHARD_DIR}"
    exit 1
fi
if ! command -v pigz &> /dev/null; then
    echo "Error: pigz is not installed. Install it and rerun the script."
    exit 1
fi

# Extract each tarball and merge into the temporary directory
echo "Merging tarballs from ${SHARD_DIR}..."
for TARFILE in ${SHARD_DIR}/shard_*.tar; do
    echo "Processing ${TARFILE} ..."
    tar -xf ${TARFILE} -C ${TEMP_DIR}
done

#Compress the merged content into the final tarball
echo "Creating final compressed tarball: ${FINAL_TARBALL}"
tar -cf - -C ${TEMP_DIR} . | pigz -p ${WORKERS} > ${FINAL_TARBALL}

echo "Cleaning up temporary files..."
rm -rf ${TEMP_DIR}
echo "Done! Final tarball: ${FINAL_TARBALL}"