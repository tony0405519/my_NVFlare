#!/bin/sh

i=1
while [ $i -le 20 ]
do 
	./submit_job.sh mnist_fedavg_stream_tb 1.0
	sleep 500
done
