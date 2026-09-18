#!/usr/bin/env bash
# Lança o treino completo desacoplado da sessão que o iniciou (nohup +
# setsid), para sobreviver mesmo se a ferramenta que disparou este script
# for encerrada. Uso: ./run_full_training.sh [epocas]
set -euo pipefail

cd "$(dirname "$0")"
source ~/venvs/smiledataai-tf/bin/activate

LOG="/mnt/c/Developer/WIA_Teeth_Semantic_Segmentation/data/results/tensorflow_training.log"
mkdir -p "$(dirname "$LOG")"

setsid nohup python train.py --epochs "${1:-200}" --batch-size 2 > "$LOG" 2>&1 < /dev/null &
echo $! > /tmp/smiledataai_train.pid
echo "Treino iniciado com PID $(cat /tmp/smiledataai_train.pid), log em $LOG"
