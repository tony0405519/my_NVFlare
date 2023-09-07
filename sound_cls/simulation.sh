#!/bin/sh

server="192.168.100.3"
hostname="aienode1"
password="05207"

datapath="/root/sound_datasets/urbansound8k/audio1/*"

for folder in $datapath
do
        file="$folder/*"
        for f in $file
        do
                sshpass -p ${password} scp $f ${hostname}@${server}:/home/aienode1/NVFlare/sound_cls/train_root/train_data
                sleep 1
        done
done

