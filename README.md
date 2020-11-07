# DeepLearning (Image Classification)

A showcase deep learning project for image classification with MNIST Dataset to simplify building and training deep learning models using Keras.

## Setup

### Requirements

This repository contains Dockerfile.

1. Install Docker following the installation guide for your platform: 
[https://docs.docker.com/engine/installation/](https://docs.docker.com/engine/installation/)

## Quick Start Guide
You have two options how you can run this project: with Docker or on local.

#### Docker: Build the Docker image and run it as a Container

1. Clone the repository and go to the path DeepLearning:
```bash
git clone https://github.com/datadima90/DeepLearning.git
cd DeepLearning
```

2. Build the docker using:
```bash
docker build -t datadima90/deeplearning:dl -f Dockerfile .
```

3. Once we've built the image, we have all the frameworks we need installed in it. Run the the docker image using:
```bash
docker run -it Dockerfile bash
```

#### Local: Creating an environment from an environment.yml file on local

Use the terminal or an Anconda Prompt for the following steps:

1. Create the enviroment from the environment.yml file:
```bash
conda env create -f environment.yml
```

2. Activate the new enviroment:
```bash
conda activate deepl_env
```

3. Verify that the new enviroment was installed correctly:
```bash
conda env list
```

### Running the Project
If docker is running or you created the enviroment from the .yml file you can start the training in terminal using:

```bash
python run.py -c configs/conv_from_config.json
```

Start Tensorboard visualization using:

```bash
tensorboard --logdir=experiments/<date>/<conv_from_config.json.exp.name>/logs
```

Visualize the learning curve of the training and validation process using:

```bash
python run_evaluate.py -c configs/conv_from_config.json
```
