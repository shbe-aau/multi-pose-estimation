export PYOPENGL_PLATFORM='egl'

wait

for OBJ_ID in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
do
	      ae_train cad_autoencoder/obj${OBJ_ID}
	      wait
done

wait
