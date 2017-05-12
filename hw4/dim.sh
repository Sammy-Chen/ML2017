#!/bin/bash

python3.4 gen.py > my_model
python3.4 dim.py $1 $2
