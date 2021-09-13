# Remove all training images
find ../output/ -wholename "*/images/*.png" -delete

# Remove all validation images except for epoch *0 and *9
for NUM in 1 2 3 4 5 6 7 8
do
    find ../output/ -wholename "*/val-images/epoch*${NUM}-batch*.png" -delete
    wait
done

# # Remove all models except for epoch *0 and *9
# for NUM in 1 2 3 4 5 6 7 8
# do
#     find ../output/ -wholename "*model-epoch*${NUM}.pt" -delete
#     wait
# done
