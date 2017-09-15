#!/bin/bash

input_file=$1

echo $input_file
python3 Q1.py << inp
$input_file
inp

