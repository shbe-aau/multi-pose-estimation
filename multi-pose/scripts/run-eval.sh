#!/bin/bash
# Arguments
OBJ_ID=$1 #e.g 10
APPROACH_NAME=$2 #e.g "sundermeyer"
TRAINED_MODEL=$3 #e.g "multi-pose/output/pose/obj19-10-poses-aug-bg-dataset/models/model-epoch20.pt"
DATA_SPLIT=$4 #e.g. "train"
DATASET=$5 #e.g. "tless" or "lm"

if test "$DATASET" = "tless"; then
    SUB_DATASET="primesense"
else
    SUB_DATASET=""
fi

# Parameters
#SHARED_FOLDER=$(realpath ~/share-to-docker)
SHARED_FOLDER=$(dirname $(readlink -f $0) | rev | cut -d'/' -f3- | rev)

# Docker commands
GENERAL_DOCKER="docker run --rm --runtime=nvidia --user=$( id -u $USER ):$( id -g $USER ) --volume=/etc/group:/etc/group:ro --volume=/etc/passwd:/etc/passwd:ro --volume=/etc/shadow:/etc/shadow:ro --volume=/etc/sudoers.d:/etc/sudoers.d:ro -v ${SHARED_FOLDER}:/shared-folder -w /shared-folder -e DISPLAY=$DISPLAY --env QT_X11_NO_MITSHM=1 --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw"
PYTORCH_DOCKER="${GENERAL_DOCKER} --env PYTHONPATH=/shared-folder/bop_toolkit:$PYTHONPATH pytorch3d_multi_pose"
AAE_DOCKER="${GENERAL_DOCKER} aae_sundermeyer"

echo "----------------------------------------------------------"
echo $(date +%T) "Object" ${OBJ_ID} "- dataset:" ${DATASET}"-"${DATA_SPLIT} "(hint: use tail -f log.out to see progress)"
echo $(date +%T) "Convert BOP dataset to pickle for easy processing"
BOP_PICKLE_PATH=${SHARED_FOLDER}/multi-pose/data/${DATASET}/output/${DATA_SPLIT}-${SUB_DATASET}/obj${OBJ_ID}/${DATASET}-${DATA_SPLIT}-obj${OBJ_ID}.p
if test -f "$BOP_PICKLE_PATH"; then
    echo $(date +%T) " - BOP dataset in pickle format already exists, skipping..."
else
    if test "$DATA_SPLIT" = "test"; then
	${PYTORCH_DOCKER} python multi-pose/utils/bop2pickle.py --datasets_path /shared-folder/multi-pose/data/ --obj_ids ${OBJ_ID} --dataset_split ${DATA_SPLIT} --dataset ${DATASET} > log.out
    else
	${PYTORCH_DOCKER} python multi-pose/utils/bop2pickle.py --datasets_path /shared-folder/multi-pose/data/ --obj_ids ${OBJ_ID} --dataset_split ${DATA_SPLIT} --dataset ${DATASET} > log.out
    fi
    wait
    echo $(date +%T) "Move pickle to BOP dataset folder"
    mkdir -p ${SHARED_FOLDER}/multi-pose/data/${DATASET}/output/${DATA_SPLIT}-${SUB_DATASET}/obj${OBJ_ID}
    mv ${SHARED_FOLDER}/${DATA_SPLIT}-obj${OBJ_ID}.p ${BOP_PICKLE_PATH}
    wait
fi

echo $(date +%T) "Eval method: ${APPROACH_NAME} on the pickled data"
if test "$APPROACH_NAME" = "sundermeyer"; then
    SM_PICKLE_PATH=${SHARED_FOLDER}/multi-pose/data/${DATASET}/output/${DATA_SPLIT}-${SUB_DATASET}/obj${OBJ_ID}/${APPROACH_NAME}-result-${DATA_SPLIT}-obj${OBJ_ID}.p
    if test -f "$SM_PICKLE_PATH"; then
	echo $(date +%T) " - pickle with results from Sundermeyers approach already exists, skipping..."
    else
	cp ${SHARED_FOLDER}/multi-pose/data/${DATASET}/output/${DATA_SPLIT}-${SUB_DATASET}/obj${OBJ_ID}/${DATASET}-${DATA_SPLIT}-obj${OBJ_ID}.p ${SM_PICKLE_PATH}
	wait
	${AAE_DOCKER} python -m auto_pose.ae.images2poses ${DATASET}_autoencoder/obj${OBJ_ID} /shared-folder/multi-pose/data/${DATASET}/output/${DATA_SPLIT}-${SUB_DATASET}/obj${OBJ_ID}/sundermeyer-result-${DATA_SPLIT}-obj${OBJ_ID}.p > log.out
	wait
	echo $(date +%T) "Convert results from Sundermeyer to CSV for BOP toolkit"
	SM_CSV_PATH=${SHARED_FOLDER}/multi-pose/data/${DATASET}/output/${DATA_SPLIT}-${SUB_DATASET}/obj${OBJ_ID}/${APPROACH_NAME}-obj${OBJ_ID}_${DATASET}-${DATA_SPLIT}-${SUB_DATASET}.csv
	if test -f "$SM_CSV_PATH"; then
	    echo $(date +%T) " - CSV with results from Sundermeyers approach already exists, skipping..."
	else
	    ${PYTORCH_DOCKER} python multi-pose/eval-pickle.py -pi /shared-folder/multi-pose/data/${DATASET}/output/${DATA_SPLIT}-${SUB_DATASET}/obj${OBJ_ID}/${APPROACH_NAME}-result-${DATA_SPLIT}-obj${OBJ_ID}.p -o /shared-folder/multi-pose/data/${DATASET}/output/${DATA_SPLIT}-${SUB_DATASET}/obj${OBJ_ID}/sundermeyer-obj${OBJ_ID}_${DATASET}-${DATA_SPLIT}-${SUB_DATASET}.csv > log.out # -op multi-pose/data/${DATASET}-obj${OBJ_ID}/cad/obj_${OBJ_ID}.ply
	    wait
	fi
    fi
else
    OUR_CSV_PATH=${SHARED_FOLDER}/multi-pose/data/${DATASET}/output/${DATA_SPLIT}-${SUB_DATASET}/obj${OBJ_ID}/${APPROACH_NAME}-obj${OBJ_ID}_${DATASET}-${DATA_SPLIT}-${SUB_DATASET}.csv
    if test -f "$OUR_CSV_PATH"; then
	echo $(date +%T) " - CSV with results already exists, skipping..."
    else
	ENCODER_WEIGHTS="multi-pose/data/encoder/obj1-18/encoder.npy"
	${PYTORCH_DOCKER} python multi-pose/eval-pickle.py -pi /shared-folder/multi-pose/data/${DATASET}/output/${DATA_SPLIT}-${SUB_DATASET}/obj${OBJ_ID}/${DATASET}-${DATA_SPLIT}-obj${OBJ_ID}.p -o /shared-folder/multi-pose/data/${DATASET}/output/${DATA_SPLIT}-${SUB_DATASET}/obj${OBJ_ID}/${APPROACH_NAME}-obj${OBJ_ID}_${DATASET}-${DATA_SPLIT}-${SUB_DATASET}.csv -ep ${ENCODER_WEIGHTS} -mp ${TRAINED_MODEL} > log.out #-op multi-pose/data/${DATASET}-obj${OBJ_ID}/cad/obj_${OBJ_ID}.ply
	wait
    fi

fi

echo $(date +%T) "Run BOP eval"
RESULT_CSV=/shared-folder/multi-pose/data/${DATASET}/output/${DATA_SPLIT}-${SUB_DATASET}/obj${OBJ_ID}/${APPROACH_NAME}-obj${OBJ_ID}_${DATASET}-${DATA_SPLIT}-${SUB_DATASET}.csv
BOP_EVAL_PATH=/shared-folder/multi-pose/data/${DATASET}/output/${DATA_SPLIT}-${SUB_DATASET}/obj${OBJ_ID}/eval
LOCAL_BOP_EVAL_PATH=${SHARED_FOLDER}/multi-pose/data/${DATASET}/output/${DATA_SPLIT}-${SUB_DATASET}/obj${OBJ_ID}/eval/${APPROACH_NAME}-obj${OBJ_ID}_${DATASET}-${DATA_SPLIT}_${SUB_DATASET}
if test -d "$LOCAL_BOP_EVAL_PATH"; then
    echo $(date +%T) " - eval dir from BOP already exists, skipping..."
else
    TARGETS_FILE="/shared-folder/multi-pose/data/tless/test-targets/${DATA_SPLIT}_targets_obj${OBJ_ID}.json"
    ${PYTORCH_DOCKER} python bop_toolkit/scripts/eval_bop19.py --result_filenames=${RESULT_CSV} --targets_filename=${TARGETS_FILE} --datasets_path /shared-folder/multi-pose/data/ --eval_path ${BOP_EVAL_PATH} > log.out
fi

echo $(date +%T) "Plot BOP performance"
${PYTORCH_DOCKER} python bop_toolkit/scripts/show_performance_bop19.py --result_filenames=${RESULT_CSV} --datasets_path /shared-folder/multi-pose/data/ --eval_path ${BOP_EVAL_PATH} > log.out
