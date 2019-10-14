#!/bin/sh

flatbuffers_repo="https://github.com/google/flatbuffers.git"
flatbuffers_dir="flatbuffers"
schema_url="https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/schema/schema.fbs"
schema="schema.fbs"
virtualenv="venv"

setup_venv () {
    # setup virtual environment and install needed python packages
    pip install virtualenv

    if ! [ -d $virtualenv ]; then
	echo "setting up virtual environment ..."
	python -m virtualenv --no-site-packages $virtualenv
    fi

    echo "installing packages"

    . $virtualenv/bin/activate
    pip install $flatbuffers_dir/python/
    pip install tensorflow
    pip install keras
    pip install numpy
    pip install pillow
}

setup_flatbuffers () {

    # download and build the flatbuffer code
    if [ -d $flatbuffers_dir ]; then
	echo "flatbuffers already installed."
    else
	git clone $flatbuffers_repo
    fi

    cd $flatbuffers_dir
    cmake -G "Unix Makefiles"
    make -j4

    cd ..

    # build schema file
    if ! [ -e $schema ]; then
	echo "downloading scheme"
	wget $schema_url -O $schema
    fi

    if ! [ -d tflite ]; then
	echo "building tflite schema"
	./$flatbuffers_dir/flatc -p $schema
    fi
}

print_usage () {
    echo "Usage: $0 [all|flatbuffers|venv]"
    echo ""
    echo "Options:"
    echo "  -h, --help   print this message"
}

if [ "$1" = "all" ]; then
    setup_flatbuffers
    setup_venv
else
    case "$1" in
	-h|--help)
	    print_usage
	    exit 0
	    ;;
	flatbuffers)
	    setup_flatbuffers
	    ;;
	venv)
	    setup_venv
	    ;;
	*)
	    print_usage
	    exit 0
	    ;;
    esac
fi

