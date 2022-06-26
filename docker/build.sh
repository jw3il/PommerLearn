#!/bin/bash

# First get the base dir of this script
# see https://stackoverflow.com/questions/242538/unix-shell-script-find-out-which-directory-the-script-file-resides

# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")

# The tag we want to build
if [[ -z "${TAG}" ]]; then
  TAG=tensorrt
  echo "Using default TAG='${TAG}'. Available:"
  find ${SCRIPTPATH}/* -type d | sed -r 's/.*\/(.*)$/- \1/'
fi

echo "Building pommer image with tag '${TAG}'."

# Trick: Insert additional build args before the path using "$@" (e.g. --no-cache)
# Build our image with the current timestamp
docker build --build-arg BUILD_TIMESTAMMP=$(date +%Y%m%d-%H%M%S) -t "pommer:${TAG}" "$@" "${SCRIPTPATH}/${TAG}/"
