#!/bin/bash

cd ./build
make -j3
cd ..

./build/bin/dso_dataset_euroc \
files=/Users/jg/Downloads/mav0 \
calib=/Users/jg/Documents/dso/configs/EuRoC/camera_left_rec.txt \
calibRight=/Users/jg/Documents/dso/configs/EuRoC/camera_right_rec.txt \
preset=0 \
mode=1
