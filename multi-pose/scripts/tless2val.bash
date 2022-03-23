#!/bin/bash

# Arguments
OBJ_ID=$1 #e.g 10
DATA_SPLIT=$2 #e.g. "train"
DATASET=$3 #e.g. "tless" or "lm"

if test "$DATASET" = "tless"; then
    SUB_DATASET="primesense"
else
    SUB_DATASET=""
fi

SHARED_FOLDER=$(dirname $(readlink -f $0) | rev | cut -d'/' -f3- | rev)

# Docker commands
GENERAL_DOCKER="docker run --rm --runtime=nvidia --user=$( id -u $USER ):$( id -g $USER ) --volume=/etc/group:/etc/group:ro --volume=/etc/passwd:/etc/passwd:ro --volume=/etc/shadow:/etc/shadow:ro --volume=/etc/sudoers.d:/etc/sudoers.d:ro -v ${SHARED_FOLDER}:/shared-folder -w /shared-folder -e DISPLAY=$DISPLAY --env QT_X11_NO_MITSHM=1 --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw"
PYTORCH_DOCKER="${GENERAL_DOCKER} --env PYTHONPATH=/shared-folder/bop_toolkit:$PYTHONPATH pytorch3d_multi_pose"

if test "$DATA_SPLIT" = "test"; then
    ${PYTORCH_DOCKER} python multi-pose/utils/bop2pickle.py --datasets_path /shared-folder/multi-pose/data/ --obj_ids ${OBJ_ID} --dataset_split ${DATA_SPLIT} --dataset ${DATASET} > log.out
else
    ${PYTORCH_DOCKER} python multi-pose/utils/bop2pickle.py --datasets_path /shared-folder/multi-pose/data/ --obj_ids ${OBJ_ID} --dataset_split ${DATA_SPLIT} --dataset ${DATASET} > log.out
fi
wait
mkdir -p ${SHARED_FOLDER}/multi-pose/data/validationsets/
mv ${SHARED_FOLDER}/${DATA_SPLIT}-obj${OBJ_ID}.p ${SHARED_FOLDER}/multi-pose/data/validationsets/${DATASET}-${DATA_SPLIT}-obj${OBJ_ID}.p
wait
