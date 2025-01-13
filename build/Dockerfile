FROM kovaleskilab/ml_basic:v4-kube

SHELL ["/bin/bash", "-c"]

# Install dependencies and download kubectl
RUN pip install transformers accelerate

WORKDIR /develop/data
WORKDIR /develop/results
WORKDIR /develop/code