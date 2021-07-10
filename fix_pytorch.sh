#!/bin/bash
# This script fixes some ambiguities in PyTorch manually (required until fixed are included in current version)
# See https://github.com/pytorch/pytorch/pull/45736
# Use Main path as input, e.g. $CONDA_ENV_PATH/lib/python3.7/site-packages/torch

if [ -z $1 ]
then
  echo "PATH not defined!"
  exit 1
fi

TORCH_LIB=$1

echo "Fixing PyTorch in Path $TORCH_LIB"

sed -i 's: optional<: c10\:\:optional<:g' "$TORCH_LIB/include/ATen/DeviceGuard.h"
sed -i 's: nullopt: c10\:\:nullopt:g' "$TORCH_LIB/include/ATen/DeviceGuard.h"

sed -i 's:(optional<:(c10\:\:optional<:g' "$TORCH_LIB/include/ATen/Functions.h"
sed -i 's: optional<: c10\:\:optional<:g' "$TORCH_LIB/include/ATen/Functions.h"

echo "Done."
