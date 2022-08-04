#!/usr/bin/env bash

set -e

if [[ -z $1 ]]; then
    echo "USAGE: $0 DATA_DIR"
    exit 1
fi

DATA_DIR=$(realpath $1)
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

echo "DATA_DIR=$DATA_DIR; SCRIPT_DIR=$SCRIPT_DIR"

# mkdir -pv $DATA_DIR

SOURCE_DATA_DIR=$DATA_DIR/CLIC
TRAIN=train
VAL=valid

# Convert ----------
FINAL_TRAIN_DIR=$DATA_DIR/train_CLIC_FRLLIC
FINAL_VAL_DIR=$DATA_DIR/validation_CLIC_FRLLIC

OUT_DIR=$DATA_DIR/discards
pushd $SCRIPT_DIR
echo "Resizing..."
python import_train_images.py $SOURCE_DATA_DIR $TRAIN \
        --out_dir_clean=$FINAL_TRAIN_DIR \
        --out_dir_discard=$OUT_DIR/discard_train
python import_train_images.py $SOURCE_DATA_DIR $VAL \
        --out_dir_clean=$FINAL_VAL_DIR \
        --out_dir_discard=$OUT_DIR/discard_val

# Update Cache ----------
CACHE_P=$DATA_DIR/cache.pkl
export PYTHONPATH=$(pwd)

echo "Updating cache $CACHE_P..."
python dataloaders/images_loader.py update $FINAL_TRAIN_DIR "$CACHE_P" --min_size 128
python dataloaders/images_loader.py update $FINAL_VAL_DIR "$CACHE_P" --min_size 128

echo "----------------------------------------"
echo "Done"
echo "To train, you MUST UPDATE configs/dl/clic.cf:"
echo ""
echo "  image_cache_pkl = '$1/cache.pkl'"
echo "  train_imgs_glob = '$(realpath $1/train_CLIC_FRLLIC)'"
echo "  val_glob = '$(realpath $1/validation_CLIC_FRLLIC)'"
echo ""
echo "----------------------------------------"
