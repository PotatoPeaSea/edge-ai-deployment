#!/bin/bash
# HOST-SIDE: push EAR demo files to the target device.
# Delegates to deploy.py which uses paramiko (no sshpass required).
exec python3 "$(dirname "$0")/deploy.py" "$@"
