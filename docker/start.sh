#!/bin/bash

docker run --rm -it -d --shm-size=64gb --gpus all \
    -v /home/matykina_ov/CRN:/home/docker_crn/CRN \
    -v /media/matykina_ov/HPR1:/home/docker_crn/HPR1 \
    -v /media/matykina_ov/data:/home/docker_crn/data \
    --name matykina_crn  crn:latest "/bin/bash"
