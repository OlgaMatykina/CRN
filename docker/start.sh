#!/bin/bash

docker run --rm -it -d --shm-size=64gb --gpus all \
    -v /media/matykina_ov/FastSSD/CRN:/home/docker_crn/CRN \
    -v /media/matykina_ov/FastSSD:/home/docker_crn/HPR3 \
    --name matykina_crn  crn:latest "/bin/bash"
