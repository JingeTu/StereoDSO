#!/usr/bin/env bash

cd build
make -j3
cd ..

./build/bin/euroc_rectify /home/jg/Documents/Datasets/EuRoC/V1_01_easy/mav0
