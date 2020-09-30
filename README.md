# DeepLearning

A deep learning project for image classification to simplify building and training deep learning models using Keras

## Setup

This repository contains Dockerfile.

### Requirements
1. Install Docker following the installation guide for your platform: 
[https://docs.docker.com/engine/installation/](https://docs.docker.com/engine/installation/)

## Quick Start Guide

Obtaining the Docker image

#### Build the Docker image locally
You can build the images locally. Also, since the GPU version is not available in Docker Hub at the moment, you'll have to follow this if you want to GPU version. Note that this will take an hour or two depending on your machine since it compiles a few libraries from scratch.

```bash
git clone https://github.com/datadima90/deeplearning.git
cd deeplearning
```

```bash
docker build -t datadima90/deeplearning:dl -f Dockerfile .
```

## Running the Docker image as a Container
Once we've built the image, we have all the frameworks we need installed in it. 

```bash
docker run -it Dockerfile bash
```

### Running the Project
If docker is running then you can start the training using:

```bash
python run.py -c configs/conv_from_config.json
```

Start Tensorboard visualization using:

```bash
tensorboard --logdir=experiments/<date>/<conv_from_config.json.exp.name>/logs
```

Visualize the learning curve of the training and validation data set using:

```bash
python run_evaluate.py -c configs/conv_from_config.json
```
