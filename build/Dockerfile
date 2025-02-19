FROM kovaleskilab/ml_basic:v4-kube

SHELL ["/bin/bash", "-c"]

# Install openssh-server
RUN apt-get update && apt-get install -y openssh-server && \
    mkdir -p /run/sshd

# Create the .ssh folder for root
RUN mkdir -p /root/.ssh

# Copy in your public key (the file containing your public key)
COPY authorized_keys /root/.ssh/authorized_keys

# Fix ownership and permissions
RUN chmod 700 /root/.ssh && \
    chmod 600 /root/.ssh/authorized_keys

# Make sure PermitRootLogin allows key-based root logins
RUN sed -i 's/^#PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config

EXPOSE 22

CMD ["/usr/sbin/sshd","-D"]

WORKDIR /develop/data
WORKDIR /develop/results
WORKDIR /develop/code