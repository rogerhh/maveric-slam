#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
OUTPUT_DIR=$SCRIPT_DIR/../outputs
IMG_DIR=/scratch/roger_hsiao/SuperPoint_SLAM/datasets/kitti/sequences/00/image_0

for i in {785..789}; do
    frame_number1=$(printf "%06d" $i)
    frame_number2=$(printf "%06d" $(($i + 1)) )

    img1=$IMG_DIR/$frame_number1.png
    img2=$IMG_DIR/$frame_number2.png
    echo $img1
    echo $img2

    outfile=$OUTPUT_DIR/transform_${frame_number1}_${frame_number2}.npy

    echo "Writing results to $outfile"

    (cd $SCRIPT_DIR/../python && python3 pairwise_pnp.py $img1 $img2 --outfile $outfile)

done
