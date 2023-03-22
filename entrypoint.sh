#!/bin/bash
set -e

if [ -v LOCAL_USER_ID ]; then
  # Create a new user with the specified UID and GI
  useradd --home /home/aloception --uid $LOCAL_USER_ID --shell /bin/bash dynamic && usermod -aG sudo dynamic && echo "dynamic ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
  BASE_USER=dynamic
else
  BASE_USER=aloception
fi

# Execute the provided command as the aloception user
exec /usr/sbin/gosu $BASE_USER "$@"
#exec "$@"