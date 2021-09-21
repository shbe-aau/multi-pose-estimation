#!/bin/bash

for OBJ_ID in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
do
    OBJ_ID_NO_LEAD=$(echo $OBJ_ID | sed 's/^0*//')
    OUTPUT_FILE="test_targets_obj${OBJ_ID}.json"
    touch ${OUTPUT_FILE}
    echo "[" > ${OUTPUT_FILE}

    input="test_targets_bop19.json"
    while IFS= read -r line
    do
	if [[ $line == *'"obj_id": '${OBJ_ID_NO_LEAD}','* ]]; then
	    #echo "It's there! ${line}"
	    echo "${line}" >> ${OUTPUT_FILE}
	fi
    done < "$input"
    truncate -s-3 ${OUTPUT_FILE}
    echo "
]" >> ${OUTPUT_FILE}
done
