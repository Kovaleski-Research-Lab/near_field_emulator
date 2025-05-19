FROM kovaleskilab/ml_ssh:latest

SHELL ["/bin/bash", "-c"]

# Install openssh-server
RUN apt-get update && pip install tensorboard seaborn

WORKDIR /develop/data
WORKDIR /develop/results
WORKDIR /develop/code