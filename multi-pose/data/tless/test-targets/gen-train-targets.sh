for OBJ_ID in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
do
    # Remove leading zeros
    OBJ_ID_NO_LEAD=$(echo $OBJ_ID | sed 's/^0*//')
    sed "s/OBJID/${OBJ_ID_NO_LEAD}/g" train_targets_template.json > train_targets_obj${OBJ_ID}.json
    wait
done

wait
