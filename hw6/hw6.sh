#!/bin/bash

if [ -f "./encoder.h5" ]; then
    # file exists
    echo "File exists"
else
    # file not exist
    echo "File does not exists."
    wget --no-check-certificate "https://www.dropbox.com/s/yzcx4s3hm6qlx7f/encoder.h5?dl=1" -O encoder.h5
fi


python cluster_.py $1 $2 $3
