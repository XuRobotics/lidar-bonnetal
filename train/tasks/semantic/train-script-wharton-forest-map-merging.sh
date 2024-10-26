#!/bin/sh
python3 train.py -d ../../../raw_training_data/ --arch_cfg ./config/arch/darknet53-1024px-pennovation.yaml --data_cfg ./config/labels/wharton-forest.yaml --log ../../../logs/ -p ""
