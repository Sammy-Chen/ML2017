#!/bin/bash

tar -zxvf my_model.tgz 
MODEL=my_model.h5
python3.5 predict.py $1 $MODEL $2 
