#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_status "Pod Setup Script Starting..."

# Step 1: Read SSH command from terminal and parse IP and port
print_status "Please enter the SSH command (e.g., 'ssh root@38.128.232.9 -p 45752 -i ~/.ssh/id_ed25519'):"
echo -n "> "
read -r ssh_command

# Parse the SSH command to extract IP and port
if [[ $ssh_command =~ ssh[[:space:]]+root@([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)[[:space:]]+-p[[:space:]]+([0-9]+) ]]; then
    SERVER_IP="${BASH_REMATCH[1]}"
    SERVER_PORT="${BASH_REMATCH[2]}"
    print_status "Extracted IP: $SERVER_IP, Port: $SERVER_PORT"
else
    print_error "Failed to parse SSH command. Expected format: ssh root@IP -p PORT -i KEYFILE"
    exit 1
fi

# Step 2 & 3: SCP scripts to remote server and make them executable
print_status "Copying setup scripts to remote server..."

# Check if the required scripts exist
if [[ ! -f "$SCRIPT_DIR/pod_root_setup.sh" ]]; then
    print_error "pod_root_setup.sh not found in $SCRIPT_DIR"
    exit 1
fi

if [[ ! -f "$SCRIPT_DIR/pod_user_setup.sh" ]]; then
    print_error "pod_user_setup.sh not found in $SCRIPT_DIR"
    exit 1
fi

# Create workspace directory on remote server and copy scripts
ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa -p "$SERVER_PORT" root@"$SERVER_IP" "mkdir -p /workspace"

scp -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa -P "$SERVER_PORT" \
    "$SCRIPT_DIR/pod_root_setup.sh" \
    "$SCRIPT_DIR/pod_user_setup.sh" \
    root@"$SERVER_IP":/workspace/

# Stage pod GitHub SSH key in /workspace (user doesn't exist yet —
# pod_root_setup.sh will move it to ~alexs/.ssh/ after creating the user)
POD_KEY="$HOME/.ssh/pod_github"
if [[ -f "$POD_KEY" ]]; then
    print_status "Staging pod GitHub SSH key in /workspace..."
    scp -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa -P "$SERVER_PORT" \
        "$POD_KEY" root@"$SERVER_IP":/workspace/pod_github
    scp -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa -P "$SERVER_PORT" \
        "${POD_KEY}.pub" root@"$SERVER_IP":/workspace/pod_github.pub
else
    print_warning "Pod GitHub key not found at $POD_KEY. Git clones may fail."
    print_warning "Generate with: ssh-keygen -t ed25519 -f ~/.ssh/pod_github -N '' -C 'gc635@runpod'"
fi

# Make scripts executable on remote server
ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa -p "$SERVER_PORT" root@"$SERVER_IP" \
    "chmod +x /workspace/pod_root_setup.sh /workspace/pod_user_setup.sh"

print_status "Scripts copied and made executable on remote server"

# Step 4: Update local ~/.ssh/config
print_status "Updating local ~/.ssh/config..."

SSH_CONFIG_FILE="$HOME/.ssh/config"

if [[ ! -f "$SSH_CONFIG_FILE" ]]; then
    print_warning "SSH config file not found, creating new one..."
    mkdir -p "$HOME/.ssh"
    touch "$SSH_CONFIG_FILE"
fi

# Check if pod host entry exists
if grep -q "^Host pod" "$SSH_CONFIG_FILE"; then
    # Update existing entry
    print_status "Updating existing 'pod' host entry in SSH config"
    
    # Use sed to update the HostName and Port for the pod host
    # Create a temporary file for the update
    temp_file=$(mktemp)
    
    # Process the config file
    awk -v ip="$SERVER_IP" -v port="$SERVER_PORT" '
    /^Host pod/ { in_pod_section = 1; print; next }
    /^Host / && !/^Host pod/ { in_pod_section = 0; print; next }
    in_pod_section && /^[[:space:]]*HostName/ { print "    HostName " ip; next }
    in_pod_section && /^[[:space:]]*Port/ { print "    Port " port; next }
    { print }
    ' "$SSH_CONFIG_FILE" > "$temp_file"
    
    mv "$temp_file" "$SSH_CONFIG_FILE"
else
    # Add new entry
    print_status "Adding new 'pod' host entry to SSH config"
    cat >> "$SSH_CONFIG_FILE" << EOF

Host pod
    HostName $SERVER_IP
    User alexs
    Port $SERVER_PORT
EOF
fi

print_status "SSH config updated successfully"

# Step 5: Run pod_root_setup.sh on remote server
print_status "Running pod_root_setup.sh on remote server..."
ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa -p "$SERVER_PORT" root@"$SERVER_IP" \
    "cd /workspace && ./pod_root_setup.sh"

print_status "Root setup completed successfully"

# Step 6: Success message
print_status "🎉 Pod setup completed successfully!"
print_status "You can now connect to your pod using: ssh pod"
print_status "Server details:"
print_status "  - IP Address: $SERVER_IP"
print_status "  - Port: $SERVER_PORT"
