#!/bin/bash
set -e

if [ -v LOCAL_USER_ID ]; then
  if id -u ${LOCAL_USER_ID} >/dev/null 2>&1; then
    BASE_USER=aloception
  else
    # Create a new user with the specified UID and GI
    useradd --home /home/aloception --uid $LOCAL_USER_ID --shell /bin/bash dynamic && usermod -aG sudo dynamic && echo "dynamic ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
    BASE_USER=dynamic
  fi
else
  BASE_USER=aloception
fi

echo "Starting with UID : $LOCAL_USER_ID, base user: $BASE_USER"

su -s /bin/bash $BASE_USER 
# Execute the provided command as the aloception user
#exec /usr/sbin/gosu $BASE_USER "$@"
exec "$@"