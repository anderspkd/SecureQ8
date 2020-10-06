#!/bin/bash

if test -z "$1" -o -z "$2" -o -z "$3" -o -z "$4" -o -z "$5"; then
    echo "Usage: $0 <model> <image> <protocol> <truncation> <n_threads>"
    exit 1
fi

model=$1
image=$2
protocol=$3
trunc=$4
n_threads=$5
shift 5
args="conv2ds cisc$*"

out_dir=MP-SPDZ/Player-Data
test -e $out_dir || mkdir $out_dir

wget -nc -q https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_${model}_quant.tgz

tar --no-same-owner -xzf mobilenet_${model}_quant.tgz

source venv/bin/activate

./model.py --model_out $out_dir/Input-P0-0 mobilenet_${model}_quant.tflite

./image_to_numbers.py mobilenet_${model}_quant.tflite $image $out_dir/Input-P1-0 2> /dev/null

cd MP-SPDZ

case $protocol in
    0) protocol=semi2k ;;
    1) protocol=hemi ;;
    2) protocol=ring ;;
    3) protocol=rep-field ;;
    4) protocol=spdz2k ;;
    5) protocol=cowgear ;;
    6) protocol=ps-rep-ring ;;
    7) protocol=ps-rep-field ;;
esac

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

if [[ $trunc = prob ]]; then
    trunc=0
elif [[ $trunc = exact ]]; then
    trunc=2
else
    echo Invalid argument for truncation, must be 'exact' or 'prob'
    exit 1
fi

if [[ $protocol =~ ring ]]; then
    args="$args split"
    if [[ $trunc = 0 ]]; then
	trunc=1
    fi
fi

python ./compile.py -D $opt benchmark_mobilenet $model $trunc $n_threads $args | grep -v WARNING

touch ~/.rnd
Scripts/setup-ssl.sh

for i in $(seq 0 $[N-1]); do
    echo $i
    echo "${hosts[$i]}"
done

opts=${args// /-}
test $opts && opts=-$opts
prog=benchmark_mobilenet-$model-$trunc-$n_threads$opts

bin=$protocol-party.x

if [[ $protocol = ring ]]; then
    bin=replicated-ring-party.x
fi
