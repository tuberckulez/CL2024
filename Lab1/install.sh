#!/bin/bash

apt update
apt upgrade -y
apt install wget -y
apt install python3 -y
apt install python3-pip -y
apt install python3-venv -y
apt install vim -y
apt install -y libsndfile1 ffmpeg
source ~/.bashrc
cd /home/user/lab1_files
python3 -m venv env
source env/bin/activate
pip3 install Cython
pip3 install nemo_toolkit['all']
pip3 install fastapi
pip3 install uvicorn
pip3 install python-multipart
pip3 install requests
mkdir data && cd data
wget https://github.com/sberdevices/golos/raw/master/examples/data/001ce26c07c20eaa0d666b824c6c6924.wav
cd ../ && mkdir models && cd models
wget https://us.openslr.org/resources/114/QuartzNet15x5_golos.nemo.gz
gzip -dv QuartzNet15x5_golos.nemo.gz
cd ../ && mkdir -p src/examples && cd src/examples
wget https://raw.githubusercontent.com/sberdevices/golos/master/examples/infer.py
