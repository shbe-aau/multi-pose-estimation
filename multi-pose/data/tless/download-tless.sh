#!/bin/bash
wget "https://ptak.felk.cvut.cz/6DB/public/bop_datasets/tless_models.zip" --no-check-certificate -O tless-models.zip
unzip tless-models.zip
rm tless-models.zip

wget "https://ptak.felk.cvut.cz/6DB/public/bop_datasets/tless_test_primesense_bop19.zip" --no-check-certificate -O tless-test-images.zip
unzip tless-test-images.zip
rm tless-test-images.zip

wget "https://ptak.felk.cvut.cz/6DB/public/bop_datasets/tless_train_primesense.zip" --no-check-certificate -O tless-train-images.zip
unzip tless-train-images.zip
rm tless-train-images.zip
