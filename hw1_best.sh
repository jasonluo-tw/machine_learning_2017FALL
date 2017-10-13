#!/bin/bash

input_file=$1
output_file=$2

echo $input_file
echo $output_file

python3 hw1_best.py << inp
$input_file
$output_file
inp
                               
