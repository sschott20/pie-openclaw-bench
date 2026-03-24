#!/bin/bash
set -e

# Set up basic package environment
yes | unminimize
apt update
apt upgrade -y
apt install -y vim zsh sudo pkg-config libssl-dev git htop less xxd

# Create user (chown handles pre-existing home dir where useradd -m skips ownership)
useradd -m alexs
chown -R alexs:alexs /home/alexs
usermod -aG sudo alexs
chsh -s /bin/zsh alexs

# Move staged SSH key to user's home (was staged in /workspace by pod_setup.sh)
if [ -f /workspace/pod_github ]; then
    mkdir -p /home/alexs/.ssh
    mv /workspace/pod_github /home/alexs/.ssh/id_ed25519
    mv /workspace/pod_github.pub /home/alexs/.ssh/id_ed25519.pub
    chmod 600 /home/alexs/.ssh/id_ed25519 /home/alexs/.ssh/id_ed25519.pub
    chown -R alexs:alexs /home/alexs/.ssh
fi

# Switch user
runuser -u alexs /workspace/pod_user_setup.sh
