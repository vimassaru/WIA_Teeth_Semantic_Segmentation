#!/usr/bin/env bash
set -euo pipefail

VENV="$HOME/venvs/smiledataai-tf"
NVDIR="$VENV/lib/python3.11/site-packages/nvidia"

if grep -q "CUDA/cuDNN libs instalados via pip" "$VENV/bin/activate"; then
  echo "Já configurado, nada a fazer."
  exit 0
fi

{
  echo ""
  echo "# CUDA/cuDNN libs instalados via pip (tensorflow[and-cuda]) nao ficam no"
  echo "# search path padrao do linker - exportamos aqui para o TF achar a GPU."
  echo "export LD_LIBRARY_PATH=\"$NVDIR/cublas/lib:$NVDIR/cuda_cupti/lib:$NVDIR/cuda_nvrtc/lib:$NVDIR/cuda_runtime/lib:$NVDIR/cudnn/lib:$NVDIR/cufft/lib:$NVDIR/curand/lib:$NVDIR/cusolver/lib:$NVDIR/cusparse/lib:$NVDIR/nccl/lib:$NVDIR/nvjitlink/lib\""
} >> "$VENV/bin/activate"

echo "Adicionado ao $VENV/bin/activate"
tail -5 "$VENV/bin/activate"
