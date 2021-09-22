#!/bin/bash
wget "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar" --no-check-certificate -O background-imgs.tar
tar xvf background-imgs.tar
rm background-imgs.tar
mv VOCdevkit/VOC2012/* .
rm -rf VOCdevkit
