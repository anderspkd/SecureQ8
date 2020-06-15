#!/bin/bash

model=$1
image=$2
protocol=$3
trunc=${4:-0}
n_threads=${5:-1}
shift 5

out_dir=MP-SPDZ/Player-Data
mkdir $out_dir

wget -nc https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_${model}_quant.tgz

tar xzf mobilenet_${model}_quant.tgz

source venv/bin/activate

./model.py --model_out $out_dir/Input-P0-0 mobilenet_${model}_quant.tflite

./image_to_numbers.py mobilenet_${model}_quant.tflite $image $out_dir/Input-P1-0

cd MP-SPDZ

if [[ $protocol =~ ring || $protocol =~ 2k ]]; then
    opt='-R 72'
    run_opt="$opt"
fi

if [[ $protocol = spdz2k ]]; then
    run_opt='-R 72 -S 48'
fi

if [[ $protocol = cowgear ]]; then
    run_opt='-l 40'
fi

python ./compile.py -D $opt benchmark_mobilenet $model $trunc $n_threads $*

Scripts/setup-ssl.sh

IFS=-
opts=$*
IFS=' '
test $opts && opts=-$opts
Scripts/$protocol.sh benchmark_mobilenet-$model-$trunc-$n_threads$opts $run_opt

# allow debugging with docker
exit 0
