bash run-eval.sh 08 "rgb-cad-depth-cad-occ-sm-pose-41epochs" "pytorch3d/output/depth/multi-obj08-test/rgb-cad-depth-cad-occ-sm-pose/models/model-epoch41.pt" "test" "tless"

bash run-eval.sh 08 "rgb-reconst-depth-cad-occ-sm-pose-41epochs" "pytorch3d/output/depth/multi-obj08-test/rgb-reconst-depth-cad-occ-sm-pose/models/model-epoch41.pt" "test" "tless"

bash run-eval.sh 08 "rgb-reconst-depth-reconst-occ-viewsphere-41epochs" "pytorch3d/output/depth/multi-obj08-test/rgb-reconst-depth-reconst-occ-viewsphere/models/model-epoch41.pt" "test" "tless"


bash run-eval.sh 23 "rgb-reconst-depth-cad-sm-pose-141epochs" "pytorch3d/output/depth/multi-path-reconst/depth-cad/obj23/models/model-epoch141.pt" "test" "tless"

bash run-eval.sh 22 "rgb-reconst-depth-reconst-sm-pose-150epochs" "pytorch3d/output/depth/multi-path-reconst/depth-reconst/obj22/models/model-epoch150.pt" "train" "tless"

bash run-eval.sh 22 "rgb-reconst-depth-reconst-sm-pose-49epochs" "pytorch3d/output/depth/multi-path-reconst/depth-reconst/obj22/models/model-epoch49.pt" "test" "tless"


bash run-eval.sh 25 "rgb-reconst-depth-reconst-10views-sm-pose-141epochs" "pytorch3d/output/depth/multi-path-views/obj25-10views/models/model-epoch141.pt" "test" "tless"

for OBJ_ID in 25 20 04 12
do
    bash run-eval.sh ${OBJ_ID} "new-occ-new-encoder-199epochs" "pytorch3d/output/depth/new-occlusions/new-encoder/obj${OBJ_ID}/models/model-epoch199.pt" "test" "tless"
    wait
    bash run-eval.sh ${OBJ_ID} "new-occ-new-encoder-99epochs" "pytorch3d/output/depth/new-occlusions/new-encoder/obj${OBJ_ID}/models/model-epoch99.pt" "test" "tless"
    wait
done

# for OBJ_ID in 25 #19 25 08
# do
#     bash run-eval.sh ${OBJ_ID} "viewsphere-aug-600views-18cyclo-49epochs" "pytorch3d/output/depth/viewsphere-aug-600views-18cyclo/obj${OBJ_ID}/models/model-epoch49.pt" "test" "tless"
#     wait
#     #bash run-eval.sh ${OBJ_ID} "6views-depth-max30-199epochs" "pytorch3d/output/depth/all-objs-max30/6views/obj${OBJ_ID}/models/model-epoch199.pt" "test" "tless"
#     #wait
# done

# #Eval all objs with union and adding pose
# for OBJ_ID in 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 09 08 07 06 05 04 03 02 01
# do
#     bash run-eval.sh ${OBJ_ID} "6views-depth-max30-199epochs" "pytorch3d/output/depth/all-objs-max30/6views/obj${OBJ_ID}/models/model-epoch199.pt" "train" "tless"
#     wait
#     bash run-eval.sh ${OBJ_ID} "6views-depth-max30-199epochs" "pytorch3d/output/depth/all-objs-max30/6views/obj${OBJ_ID}/models/model-epoch199.pt" "test" "tless"
#     wait
# done

# #Eval all objs with union and adding pose
# for OBJ_ID in 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 09 08 07 06 05 04 03 02 01
# do
#     bash run-eval.sh ${OBJ_ID} "all-objs-vsd-union-add-99epochs" "pytorch3d/output/depth/all-objs-vsd-union-add/obj${OBJ_ID}/models/model-epoch99.pt" "train" "tless"
#     wait
#     bash run-eval.sh ${OBJ_ID} "all-objs-vsd-union-add-99epochs" "pytorch3d/output/depth/all-objs-vsd-union-add/obj${OBJ_ID}/models/model-epoch99.pt" "test" "tless"
#     wait
#     bash run-eval.sh ${OBJ_ID} "sundermeyer" "" "train" "tless"
#     wait
#     bash run-eval.sh ${OBJ_ID} "sundermeyer" "" "test" "tless"
#     wait
# done

# for OBJ_ID in 25 19 12
# do
#     bash run-eval.sh ${OBJ_ID} "sm-pose-reuse-depth-max30-99epochs" "pytorch3d/output/depth/all-objs-max30/sm-pose-reuse/obj${OBJ_ID}/models/model-epoch99.pt" "train" "tless"
#     wait
#     #bash run-eval.sh ${OBJ_ID} "sm-pose-reuse-depth-max30-99epochs" "pytorch3d/output/depth/all-objs-max30/sm-pose-reuse/obj${OBJ_ID}/models/model-epoch99.pt" "test" "tless"
#     #wait
# done

# for OBJ_ID in 25 19 12 06 04
# do
#     bash run-eval.sh ${OBJ_ID} "sm-pose-depth-max30-99epochs" "pytorch3d/output/depth/all-objs-max30/sm-pose/obj${OBJ_ID}/models/model-epoch99.pt" "train" "tless"
#     wait
#     #bash run-eval.sh ${OBJ_ID} "sm-pose-depth-max30-99epochs" "pytorch3d/output/depth/all-objs-max30/sm-pose/obj${OBJ_ID}/models/model-epoch99.pt" "test" "tless"
#     wait
#     bash run-eval.sh ${OBJ_ID} "noise-depth-max30-99epochs" "pytorch3d/output/depth/all-objs-max30/noise/obj${OBJ_ID}/models/model-epoch99.pt" "train" "tless"
#     wait
#     #bash run-eval.sh ${OBJ_ID} "noise-depth-max30-99epochs" "pytorch3d/output/depth/all-objs-max30/noise/obj${OBJ_ID}/models/model-epoch99.pt" "test" "tless"
#     wait
#     bash run-eval.sh ${OBJ_ID} "6views-depth-max30-99epochs" "pytorch3d/output/depth/all-objs-max30/6views/obj${OBJ_ID}/models/model-epoch99.pt" "train" "tless"
#     wait
#     #bash run-eval.sh ${OBJ_ID} "6views-depth-max30-99epochs" "pytorch3d/output/depth/all-objs-max30/6views/obj${OBJ_ID}/models/model-epoch99.pt" "test" "tless"
#     wait
#     bash run-eval.sh ${OBJ_ID} "big-occ-depth-max30-99epochs" "pytorch3d/output/depth/all-objs-max30/big-occ/obj${OBJ_ID}/models/model-epoch99.pt" "train" "tless"
#     wait
#     #bash run-eval.sh ${OBJ_ID} "big-occ-depth-max30-99epochs" "pytorch3d/output/depth/all-objs-max30/big-occ/obj${OBJ_ID}/models/model-epoch99.pt" "test" "tless"
#     wait
#     bash run-eval.sh ${OBJ_ID} "vis-mask-depth-max30-99epochs" "pytorch3d/output/depth/all-objs-max30/vis-mask/obj${OBJ_ID}/models/model-epoch99.pt" "train" "tless"
#     wait
#     #bash run-eval.sh ${OBJ_ID} "vis-mask-depth-max30-99epochs" "pytorch3d/output/depth/all-objs-max30/vis-mask/obj${OBJ_ID}/models/model-epoch99.pt" "test" "tless"
#     wait
# done

# bash run-eval.sh 25 "sm-pose-99epochs" "pytorch3d/output/depth/sm-pose/obj25/models/model-epoch99.pt" "train" "tless"
# wait
# bash run-eval.sh 25 "sm-pose-99epochs" "pytorch3d/output/depth/sm-pose/obj25/models/model-epoch99.pt" "test" "tless"
# wait

# # Different networks
# for OBJ_ID in 25 # 06
# do
#     bash run-eval.sh ${OBJ_ID} "confs-loss-test-99epochs" "pytorch3d/output/depth/confs-loss-test/obj${OBJ_ID}/models/model-epoch99.pt" "train" "tless"
#     wait
#     bash run-eval.sh ${OBJ_ID} "confs-loss-test-99epochs" "pytorch3d/output/depth/confs-loss-test/obj${OBJ_ID}/models/model-epoch99.pt" "test" "tless"
#     wait
# done


# # Different networks
# for OBJ_ID in 25 #04 #07 #19 12 25 06
# do
#     MODEL="skip-only-four-extra"
#     bash run-eval.sh ${OBJ_ID} "${MODEL}-30epochs" "pytorch3d/output/depth/network-test/${MODEL}/obj${OBJ_ID}/models/model-epoch30.pt" "train" "tless"
#     wait
#     bash run-eval.sh ${OBJ_ID} "${MODEL}-30epochs" "pytorch3d/output/depth/network-test/${MODEL}/obj${OBJ_ID}/models/model-epoch30.pt" "test" "tless"
#     wait
# done

# #Max depth test
# for OBJ_ID in 19 12 25 06 04 07
# do
#     for MAX_DEPTH in 10 15 30 50
#     do
# 	bash run-eval.sh ${OBJ_ID} "depth-max${MAX_DEPTH}-99epochs" "pytorch3d/output/depth/depth-max-test/max${MAX_DEPTH}/obj${OBJ_ID}/models/model-epoch99.pt" "train" "tless"
# 	wait
# 	bash run-eval.sh ${OBJ_ID} "depth-max${MAX_DEPTH}-99epochs" "pytorch3d/output/depth/depth-max-test/max${MAX_DEPTH}/obj${OBJ_ID}/models/model-epoch99.pt" "test" "tless"
# 	wait
#     done
# done


#bash run-eval.sh 19 "big-99epochs" "pytorch3d/output/depth/network-test/big/obj19/models/model-epoch99.pt" "test" "tless"
#wait
#bash run-eval.sh 19 "big-99epochs" "pytorch3d/output/depth/network-test/big/obj19/models/model-epoch99.pt" "train" "tless"
#wait


# bash run-eval.sh 20 "test-scale-99epochs" "pytorch3d/output/depth/test-scale-obj20/models/model-epoch99.pt" "test" "tless"
# wait
# bash run-eval.sh 20 "test-scale-99epochs" "pytorch3d/output/depth/test-scale-obj20/models/model-epoch99.pt" "train" "tless"
# wait

# bash run-eval.sh 19 "all-objs-vsd-union-deep-170epochs" "pytorch3d/output/depth/all-objs-vsd-union-deep/obj19/models/model-epoch170.pt" "test" "tless"
# wait
# bash run-eval.sh 19 "all-objs-vsd-union-deep-170epochs" "pytorch3d/output/depth/all-objs-vsd-union-deep/obj19/models/model-epoch170.pt" "train" "tless"
# wait
# bash run-eval.sh 12 "all-objs-vsd-union-deep-170epochs" "pytorch3d/output/depth/all-objs-vsd-union-deep/obj12/models/model-epoch170.pt" "test" "tless"
# wait
# bash run-eval.sh 12 "all-objs-vsd-union-deep-170epochs" "pytorch3d/output/depth/all-objs-vsd-union-deep/obj12/models/model-epoch170.pt" "train" "tless"
# wait

# bash run-eval.sh 20 "test-occ-120epochs" "pytorch3d/output/depth/test-occ-obj20/models/model-epoch120.pt" "test" "tless"
# wait
# bash run-eval.sh 20 "test-occ-120epochs" "pytorch3d/output/depth/test-occ-obj20/models/model-epoch120.pt" "train" "tless"
# wait

# #Eval all objs with union loss from paper
# for OBJ_ID in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
# do
#     bash run-eval.sh ${OBJ_ID} "all-objs-vsd-union-99epochs" "pytorch3d/output/depth/all-objs-vsd-union/obj${OBJ_ID}/models/model-epoch99.pt" "train" "tless"
#     wait
#     bash run-eval.sh ${OBJ_ID} "all-objs-vsd-union-99epochs" "pytorch3d/output/depth/all-objs-vsd-union/obj${OBJ_ID}/models/model-epoch99.pt" "test" "tless"
#     wait
# done

# #Eval all objs with union and adding pose
# for OBJ_ID in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
# do
#     bash run-eval.sh ${OBJ_ID} "all-objs-vsd-union-add-99epochs" "pytorch3d/output/depth/all-objs-vsd-union-add/obj${OBJ_ID}/models/model-epoch99.pt" "train" "tless"
#     wait
#     bash run-eval.sh ${OBJ_ID} "all-objs-vsd-union-add-99epochs" "pytorch3d/output/depth/all-objs-vsd-union-add/obj${OBJ_ID}/models/model-epoch99.pt" "test" "tless"
#     wait
# done

# #Eval all objs with union and small model
# for OBJ_ID in 06 07 09 #01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
# do
#     bash run-eval.sh ${OBJ_ID} "all-objs-vsd-union-small-99epochs" "pytorch3d/output/depth/all-objs-vsd-union-small/obj${OBJ_ID}/models/model-epoch99.pt" "train" "tless"
#     wait
#     bash run-eval.sh ${OBJ_ID} "all-objs-vsd-union-small-99epochs" "pytorch3d/output/depth/all-objs-vsd-union-small/obj${OBJ_ID}/models/model-epoch99.pt" "test" "tless"
#     wait
# done

# #Eval 2nd run of all objects with MSPD and MSSD
# for OBJ_ID in 25 #10 17 19
# do
#     bash run-eval.sh ${OBJ_ID} "add-pose-99epochs" "pytorch3d/output/depth/obj25-add-pose/models/model-epoch99.pt" "train" "tless"
#     wait
#     bash run-eval.sh ${OBJ_ID} "add-pose-99epochs" "pytorch3d/output/depth/obj25-add-pose/models/model-epoch99.pt" "test" "tless"
#     wait
# done

# #Eval 2nd run of all objects with MSPD and MSSD
# for OBJ_ID in 01 02 #03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
# do
#     bash run-eval.sh ${OBJ_ID} "all-objs-2nd-run-metrics-99epochs" "pytorch3d/output/depth/all-objs-2nd-run/obj${OBJ_ID}/models/model-epoch99.pt" "train" "tless"
#     wait
#     bash run-eval.sh ${OBJ_ID} "all-objs-2nd-run-metrics-99epochs" "pytorch3d/output/depth/all-objs-2nd-run/obj${OBJ_ID}/models/model-epoch99.pt" "test" "tless"
#     wait
#     bash run-eval.sh ${OBJ_ID} "sundermeyer" "" "train" "tless"
#     wait
#     bash run-eval.sh ${OBJ_ID} "sundermeyer" "" "test" "tless"
#     wait
# done

# #Eval vsd intersection vs union
# for OBJ_ID in 10 17 19
# do
#     bash run-eval.sh ${OBJ_ID} "union-max20-60epochs" "pytorch3d/output/depth/loss-test-paper/union-max20/obj${OBJ_ID}/models/model-epoch60.pt" "test" "tless"
#     bash run-eval.sh ${OBJ_ID} "intersection-max20-60epochs" "pytorch3d/output/depth/loss-test-paper/intersection-max20/obj${OBJ_ID}/models/model-epoch60.pt" "test" "tless"
#     bash run-eval.sh ${OBJ_ID} "paper-vsd-log-60epochs" "pytorch3d/output/depth/loss-test-paper/paper-vsd-log/obj${OBJ_ID}/models/model-epoch60.pt" "test" "tless"
#     bash run-eval.sh ${OBJ_ID} "paper-vsd-max20-60epochs" "pytorch3d/output/depth/loss-test-paper/paper-vsd-max20/obj${OBJ_ID}/models/model-epoch60.pt" "test" "tless"
#     bash run-eval.sh ${OBJ_ID} "all-objs-2nd-run-60epochs" "pytorch3d/output/depth/all-objs-2nd-run/obj${OBJ_ID}/models/model-epoch60.pt" "test" "tless"
# done

# #Eval linemod objects
# for OBJ_ID in 05 # 03 05 10 #14 15
# do
#     bash run-eval.sh ${OBJ_ID} "sundermeyer" "" "train" "lm"
#     wait
#     bash run-eval.sh ${OBJ_ID} "sundermeyer" "" "test" "lm"
#     wait
#     bash run-eval.sh ${OBJ_ID} "nn-130epochs" "pytorch3d/output/depth/linemod/obj${OBJ_ID}/models/model-epoch130.pt" "train" "lm"
#     wait
#     bash run-eval.sh ${OBJ_ID} "nn-130epochs" "pytorch3d/output/depth/linemod/obj${OBJ_ID}/models/model-epoch130.pt" "test" "lm"
#     wait
#     bash run-eval.sh ${OBJ_ID} "l2-pose-99epochs" "pytorch3d/output/pose/linemod/obj${OBJ_ID}/models/model-epoch99.pt" "train" "lm"
#     wait
#     bash run-eval.sh ${OBJ_ID} "l2-pose-99epochs" "pytorch3d/output/pose/linemod/obj${OBJ_ID}/models/model-epoch99.pt" "test" "lm"
#     wait
# done

# #Eval run of all objects using a small model
# for OBJ_ID in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
# do
#     bash run-eval.sh ${OBJ_ID} "all-objs-small-99epochs" "pytorch3d/output/depth/all-objs-small/obj${OBJ_ID}/models/model-epoch99.pt" "train"
#     wait
#     bash run-eval.sh ${OBJ_ID} "all-objs-small-99epochs" "pytorch3d/output/depth/all-objs-small/obj${OBJ_ID}/models/model-epoch99.pt" "test"
#     wait
# done

# #Eval 2nd run of all objects
# for OBJ_ID in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
# do
#     bash run-eval.sh ${OBJ_ID} "all-objs-2nd-run-99epochs" "pytorch3d/output/depth/all-objs-2nd-run/obj${OBJ_ID}/models/model-epoch99.pt" "train"
#     wait
#     bash run-eval.sh ${OBJ_ID} "all-objs-2nd-run-99epochs" "pytorch3d/output/depth/all-objs-2nd-run/obj${OBJ_ID}/models/model-epoch99.pt" "test"
#     wait
# done

# # Eval L2 pose loss
# for OBJ_ID in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
# do
#     bash run-eval.sh ${OBJ_ID} "l2-pose-100epochs" "pytorch3d/output/pose/all-objs/l2-pose/obj${OBJ_ID}/models/model-epoch100.pt" "train"
#     wait
#     bash run-eval.sh ${OBJ_ID} "l2-pose-100epochs" "pytorch3d/output/pose/all-objs/l2-pose/obj${OBJ_ID}/models/model-epoch100.pt" "test"
#     wait
# done


# # Eval trace pose loss
# for OBJ_ID in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
# do
#     bash run-eval.sh ${OBJ_ID} "trace-pose-100epochs" "pytorch3d/output/pose/all-objs/trace-pose/obj${OBJ_ID}/models/model-epoch100.pt" "train"
#     wait
#     bash run-eval.sh ${OBJ_ID} "trace-pose-100epochs" "pytorch3d/output/pose/all-objs/trace-pose/obj${OBJ_ID}/models/model-epoch100.pt" "test"
#     wait
# done
