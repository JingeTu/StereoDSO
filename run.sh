#!/bin/bash

cd ./build
cmake ..
make -j3
cd ..

GLOG_minloglevel=2 ./build/bin/dso_dataset_euroc \
files=/home/jingetu/Documents/EuRoC/MH_01_easy/mav0 \
calib=/home/jingetu/Desktop/workspace/mine/StereoDSO/configs/EuRoC/camera_left_rec.txt \
calibRight=/home/jingetu/Desktop/workspace/mine/StereoDSO/configs/EuRoC/camera_right_rec.txt \
preset=0 \
mode=1
