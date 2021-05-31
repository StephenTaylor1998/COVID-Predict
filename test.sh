#!/bin/bash

# train
python main.py --data-path ./datasets/china.csv


python main.py --data-path ./datasets/china.csv --test --resume weights/cnn/epoch30.pth

