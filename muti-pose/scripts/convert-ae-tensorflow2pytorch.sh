# Spin up:
# docker run -it -v ~/share-to-docker:/shared-folder --network host mmdnn/mmdnn:cpu.small

for OBJ_ID in 04 08 12 20 25 #01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
do
    cd /shared-folder/autoencoder_ws/experiments/new_occ_autoencoder/obj${OBJ_ID}
    mkdir pytorch
    cd pytorch

    mmtoir -f tensorflow -n ../checkpoints/chkpt-40000.meta -w ../checkpoints/chkpt-40000 --dstNode obj${OBJ_ID}/dense/BiasAdd -o converted
    wait

    mmtocode -f pytorch -n converted.pb -w converted.npy -d converted_pytorch.py -dw converted_pytorch.npy
    wait

    cp converted_pytorch.npy /shared-folder/autoencoder_ws/experiments/new_occ_autoencoder/obj${OBJ_ID}/obj${OBJ_ID}-encoder.npy
    wait
done

wait
