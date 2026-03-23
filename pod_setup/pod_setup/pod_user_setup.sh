#!/bin/bash
set -e
cd ~

# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
rustup target add wasm32-wasip2

# Install Oh-My-Zsh
echo Y | sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Presist Zsh history
cat >> ~/.zshrc << '__EOF__'
# --- persistent history location ---
export HISTFILE="/workspace/.zsh_history"
touch "$HISTFILE"  # create if missing

# --- sizes ---
export HISTSIZE=1000000
export SAVEHIST=1000000

# --- sane history options ---
setopt APPEND_HISTORY        # don't clobber history file, append to it
setopt INC_APPEND_HISTORY    # append right away, not only on exit
setopt SHARE_HISTORY         # merge across concurrent shells
setopt EXTENDED_HISTORY      # timestamps & durations
setopt HIST_IGNORE_SPACE     # commands starting with space aren't saved
setopt HIST_IGNORE_ALL_DUPS  # drop older duplicates on save
setopt HIST_FIND_NO_DUPS     # skip duplicates when searching

# --- continuously sync/merge ---
# On each prompt: append new lines, then read in lines from other shells
precmd() {
  setopt localoptions noxtrace # don't print commands run in this function
  { builtin history -a; builtin history -n; } >/dev/null 2>&1
}
# On shell exit: one last flush
zshexit() {
  setopt localoptions noxtrace # don't print commands run in this function
  { builtin history -a } >/dev/null 2>&1
}
__EOF__

# Install zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions)/' ~/.zshrc

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Set up SSH keys
mkdir -p ~/.ssh
cat > ~/.ssh/authorized_keys << __EOF__
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDvgZjqPCqb/07QHba7tv4GfFVVAIFHRdfWpwvQLcrDadI3IcrUTLqHkMNY+LLPKs8bKgQ1gZNi3QYyrmMvSidafuvYMlZNL1ecF9F/BQ9FlUukYnMNTXwzw9ItnbvBuHhFuG7vplGnhXKEd8+Ldt2DFHxJdE7tWcb9h4n3dia2ywE0xx+r/6uNT+rOGY4H8tI3CLoxGxK/37KZK5Qj2927tRE375pIHKYUDqI/IZeun2iGThMERx3HbxuvA7PZU0bKc5kjFPrsolpdoogOyrOv+m1CqKunZ5ppkh5dqjIfI/jN1Of1SivcgIovRdDGoUhhvJwU2RhlYg9LD0CrAtNfjMeq0BTQeigqPX2amRZNE0S6DeuIDD1u+tMSGI1e1EXhfSuP2d9p2QEs3yMGbfX72xXdwilwvpDNCTXvqFDlKDaKmu4esTzfSm4vT7/Nj/fJnSFHHfBLveQ1DnD9HpPE+SxMIHhnZNgMdiTXNlrcwXOGnZ32KzEDYAozzWoWPfM= gc635@guojun-server-1
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPcvhsYYBbgNFlCk/b6A7MDWfmlTB4bGRCzzwiz63OQK gc635@guojun-server-1
__EOF__

# Copy the pod GitHub SSH key (must be pre-generated and added to GitHub)
# Generate with: ssh-keygen -t ed25519 -f ~/.ssh/pod_github -N "" -C "gc635@runpod"
# Add to GitHub with: gh ssh-key add ~/.ssh/pod_github.pub --title "runpod-pie"
# The private key file (~/.ssh/pod_github) must exist on your local machine.
# This script expects it to be SCP'd separately or pasted manually.
if [ ! -f ~/.ssh/id_ed25519 ]; then
    echo "WARNING: ~/.ssh/id_ed25519 not found on pod. Git clone will fail."
    echo "SCP your pod_github key: scp -P PORT ~/.ssh/pod_github pod:~/.ssh/id_ed25519"
fi

chmod 600 ~/.ssh/id_ed25519 2>/dev/null || true
chmod 600 ~/.ssh/id_ed25519.pub 2>/dev/null || true

# Set up workspace
mkdir -p ~/work
cd ~/work
GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i ~/.ssh/id_ed25519" git clone git@github.com:pie-project/pie.git
GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i ~/.ssh/id_ed25519" git clone git@github.com:pie-project/model-index.git
GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i ~/.ssh/id_ed25519" git clone git@github.com:pie-project/ztensor.git

# Set up PIE (Python package with Rust backend)
cd ~/work/pie/pie
uv sync --extra cu128

# Build std inferlets (text-completion for smoke testing)
cd ~/work/pie/std/text-completion
cargo build --target wasm32-wasip2 --release

# Initialize PIE config (will prompt for model on first `pie serve`)
cd ~/work/pie/pie
uv run pie config init
uv run pie auth add $(whoami) < ~/.ssh/id_ed25519.pub

# Clone and set up benchmark repo
cd ~/work
git clone https://github.com/sschott20/pie-openclaw-bench.git pie_openclaw
cd ~/work/pie_openclaw
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install ~/work/pie/client/python/

# Build custom inferlets
cd ~/work/pie_openclaw/inferlet
cargo build --target wasm32-wasip2 --release
cd ~/work/pie_openclaw/inferlet-baseline
cargo build --target wasm32-wasip2 --release

# Install vLLM for baseline comparison
cd ~/work/pie_openclaw
source .venv/bin/activate
uv pip install vllm

# Add cargo and uv to zsh PATH
cat >> ~/.zshrc << '__PATHEOF__'
source "$HOME/.cargo/env"
source "$HOME/.local/bin/env"
__PATHEOF__
