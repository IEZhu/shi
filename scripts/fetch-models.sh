#!/usr/bin/env bash
# Download the models the pipeline needs into ./models.
#
# The app has a model manager that does this from the UI; this script exists so
# that tests and a fresh checkout can be set up without launching anything.
#
# Kept out of git: the recogniser alone is 643 MB. Tests that need it skip
# themselves when it is absent rather than failing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="${ROOT}/models"
BASE="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
RECOGNISER="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"

mkdir -p "${MODELS}"

if [ ! -f "${MODELS}/silero_vad.onnx" ]; then
  echo "fetching Silero VAD (0.6 MB)"
  curl -fL --retry 3 -o "${MODELS}/silero_vad.onnx" "${BASE}/silero_vad.onnx"
fi

if [ ! -d "${MODELS}/${RECOGNISER}" ]; then
  echo "fetching Parakeet TDT 0.6B v3 int8 (464 MB compressed, 643 MB unpacked)"
  curl -fL --retry 3 -C - -o "${MODELS}/${RECOGNISER}.tar.bz2" "${BASE}/${RECOGNISER}.tar.bz2"
  tar xjf "${MODELS}/${RECOGNISER}.tar.bz2" -C "${MODELS}"
  rm -f "${MODELS}/${RECOGNISER}.tar.bz2"
fi

echo "models ready in ${MODELS}"
