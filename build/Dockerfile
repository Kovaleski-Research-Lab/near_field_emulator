FROM kovaleskilab/ml_basic:v4-kube

SHELL ["/bin/bash", "-c"]

# Install openssh-server
RUN apt-get update && apt-get install -y openssh-server && \
    mkdir -p /run/sshd 

# Create a non-root user
RUN useradd -m -s /bin/bash devuser

# Set up SSH key auth for "devuser". 
COPY authorized_keys /home/devuser/.ssh/authorized_keys
RUN chown devuser:devuser /home/devuser/.ssh/authorized_keys && \
    chmod 600 /home/devuser/.ssh/authorized_keys

# Make sure devuser owns its .ssh directory
RUN chown -R devuser:devuser /home/devuser/.ssh && chmod 700 /home/devuser/.ssh

# Expose ssh port
EXPOSE 22

# Start sshd by default
CMD ["/usr/sbin/sshd", "-D"]

WORKDIR /develop/data
WORKDIR /develop/results
WORKDIR /develop/code