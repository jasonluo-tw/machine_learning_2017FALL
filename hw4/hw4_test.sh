#!/bin/bash

if [ -f "./weights03-0.81.hdf5" ]; then
    # file exists
    echo "File exists"
else
    # file not exist
    echo "File does not exists."
    wget --no-check-certificate "https://www.dropbox.com/s/hgqff868k6e4x5x/weights03-0.81.hdf5?dl=1" -O weights03-0.81.hdf5
    wget --no-check-certificate "https://www.dropbox.com/s/73ytmdy1wng8ffi/w2v_size128.model.bin?dl=1" -O w2v_size128.model.bin
fi


python hw4_RNN_predict.py $1 $2
