#!/bin/bash

if [ -f "./CNN_model_hw.h5" ]; then
    # file exists
    echo "File /CNN_model_hw.h5 exists"
else
    # file not exist
    echo "File /CNN_model_hw.h5 does not exists."
    wget --no-check-certificate "https://www.dropbox.com/s/3ev7sxnimo8knlm/CNN_model_hw.h5?dl=1" -O CNN_model_hw.h5

fi

python CNN_predict.py $1 $2
