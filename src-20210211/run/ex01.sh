#!/bin/bash
CASE=01
DATA=v1
SRCPATH=$1
DSTPATH=$2

function run() {
    echo "--- FOLD=${1} ---"
    python train.py \
    --train_csv=csvs/${DATA}/k${1}_train.csv \
    --valid_csv=csvs/${DATA}/k${1}_valid.csv \
    --image_dir=${SRCPATH}/images \
    --output_dir=${DSTPATH}/fold${1}/ \
    --epoch=25 \
    --scheduler=sc1:1000 \
    --image_size=512 \
    --train_augs=ex2 \
    --undersample=4000
}

if [ $# -le 2 ]; then
    KS=1
    KS=5
elif [ $# -le 3 ]; then
    KS=$3
    KE=$3
else
    KS=$3
    KE=$4
fi
for i in `seq $KS $KE`
do
    FLD=`expr $i - 1`
    run $FLD
done
