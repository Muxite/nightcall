#!/bin/sh
set -e
cd /workspace || exit 1
if [ "$#" -gt 0 ]; then
  exec "$@"
fi
exec python3 python/training/train.py
