if [ -f "./weights03-0.81.hdf5" ]; then
    # file exists
    echo "File exists"
else
    # file not exist
    echo "File does not exists."
    wget --no-check-certificate "https://www.dropbox.com/s/gi4x2bz5qe0nn3s/Model_MF500.h5?dl=1" -O Model_MF500.h5
fi

python predict.py $1 $2 $3 $4
