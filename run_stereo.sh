#!/bin/bash

cd ./build
make -j3
cd ..

./build/bin/dso_dataset \
    files=/home/jg/Documents/Datasets/Kitti/02 \
    calib=/home/jg/Documents/Datasets/Kitti/02/camera.txt \
    groundtruth=/home/jg/Documents/Datasets/Kitti/data_odometry_poses/dataset/poses/02.txt \
    preset=0 \
    mode=1
