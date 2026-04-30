#!/usr/bin/env bash
# Compile et exécute op-policy-infer — aucune dépendance Python.
# Usage : ./infer.sh <bundle_dir> <obs_csv> [--mask ...]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
swift build -c release
ARCH="$(uname -m)"
case "${ARCH}" in
  arm64) TRIPLE="arm64-apple-macosx" ;;
  x86_64) TRIPLE="x86_64-apple-macosx" ;;
  *) echo "infer.sh: architecture non supportée: ${ARCH}" >&2; exit 1 ;;
esac
exec "${HERE}/.build/${TRIPLE}/release/op-policy-infer" "$@"
