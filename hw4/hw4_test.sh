#!/bin/bash

if [ -f "./weights03-0.81.hdf5" ]; then
    # file exists
    echo "File exists"
else
    # file not exist
    echo "File does not exists."
    wget --no-check-certificate "https://www.dropbox.com/s/hgqff868k6e4x5x/weights03-0.81.hdf5?dl=1" -O weights03-0.81.hdf5

fi


python hw4_RNN_predict.py $1 $2
