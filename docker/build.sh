#!/bin/bash

echo "Building container"
docker build . \
    -f Dockerfile \
    -t crn:latest \
    --progress plain 

