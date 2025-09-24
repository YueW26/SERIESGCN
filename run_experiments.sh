#!/bin/bash
set -e

# ========= 选择要跑的实验（0=全部；1=Baseline；2=幂律；3=MixPropDual；4=Chebyshev；5=无对角）=========
EXP_ID=${EXP_ID:-6}

# ========= 基本 / 训练设置 =========
DEVICE=${DEVICE:-cuda:0}
EPOCHS=${EPOCHS:-5}
ADJTYPE=${ADJTYPE:-doubletransition}
PRINT_EVERY=${PRINT_EVERY:-50}

# ========= wandb 设置 =========
export WANDB_PROJECT=${WANDB_PROJECT:-GraphWaveNet}
export WANDB_ENTITY=${WANDB_ENTITY:-}
export WANDB_MODE=${WANDB_MODE:-online}       # online/offline
export WANDB_DIR=${WANDB_DIR:-./wandb_runs}
mkdir -p "$WANDB_DIR"

# ========= 结果表（CSV）路径 =========
export RESULTS_CSV=${RESULTS_CSV:-./results.csv}

# ========= 环境开关（依然可用于 ablation，但不再画邻接图）=========
export GWN_DIAG_MODE=${GWN_DIAG_MODE:-self_and_neighbor}  # neighbor/self_and_neighbor

# ========= 网格（可用环境变量覆盖）=========
# 支持对 DATA 与 BATCH 做网格；若未显式给 DATA_LIST/BATCH_LIST，则回退到 DATA/BATCH，再回退到默认
DATA_LIST=(${DATA_LIST:-${DATA:-data/FRANCE}})
BATCH_LIST=(${BATCH_LIST:-${BATCH:-64}})

SEQ_LIST=(${SEQ_LIST:-3})                # 3 6 12 24
PRED_LIST=(${PRED_LIST:-3})              # 3 6 12 24
LR_LIST=(${LR_LIST:-0.001 0.0001 0.00001})
DROPOUT_LIST=(${DROPOUT_LIST:-0.3})
NHID_LIST=(${NHID_LIST:-64})
WD_LIST=(${WD_LIST:-0.0001})


# ========= 跑一个实验（不再调用 _viz_probe.py）=========
run_one () {
  local EXP_GROUP="$1"   # Baseline / PowerLaw / MixPropDual / Chebyshev / NoDiagonal
  local SEQ="$2"; local PRED="$3"; local LR="$4"; local DROPOUT="$5"; local NHID="$6"; local WD="$7"
  local EXP_NAME="${EXP_GROUP}_data$(basename "$DATA")_bs${BATCH}_seq${SEQ}_pred${PRED}_lr${LR}_do${DROPOUT}_hid${NHID}_wd${WD}"

  # 训练配置（带默认兜底，避免空值导致 JSON 解析失败）
  local CFG_JSON
  CFG_JSON=$(cat <<JSON
{"exp_group":"$EXP_GROUP",
 "data":"${DATA:-data/FRANCE}","device":"${DEVICE:-cuda:0}","epochs":${EPOCHS:-5},"batch_size":${BATCH:-64},
 "seq_length":${SEQ:-12},"pred_length":${PRED:-12},"learning_rate":${LR:-0.001},"dropout":${DROPOUT:-0.3},
 "nhid":${NHID:-64},"weight_decay":${WD:-0.0001},"adjtype":"${ADJTYPE:-doubletransition}",
 "gcn_bool":true,"addaptadj":true,"randomadj":true,"print_every":${PRINT_EVERY:-50}}
JSON
)

  local CMD="python train.py \
    --data $DATA --device $DEVICE --batch_size $BATCH --epochs $EPOCHS \
    --seq_length $SEQ --pred_length $PRED \
    --learning_rate $LR --dropout $DROPOUT --nhid $NHID \
    --weight_decay $WD --print_every $PRINT_EVERY \
    --gcn_bool --addaptadj --randomadj --adjtype $ADJTYPE"

  echo ">>> [$EXP_NAME]"
  # echo "[CFG] $CFG_JSON"   # 若需排错可取消注释
  python _wandb_proxy.py --project "$WANDB_PROJECT" --name "$EXP_NAME" --config "$CFG_JSON" --cmd "$CMD"
}

# ======================== 实验开关（按 EXP_ID 选择） ========================

# ---- 实验 1：Baseline ----
if [[ $EXP_ID -eq 0 || $EXP_ID -eq 1 ]]; then
  echo "==> EXP 1: Baseline"
  export GWN_USE_POWER=0
  export GWN_USE_CHEBY=0
  export GWN_DIAG_MODE=self_and_neighbor
  for DATA in "${DATA_LIST[@]}"; do
    for BATCH in "${BATCH_LIST[@]}"; do
      # Baseline 使用 SEQ/PRED 成对（与原脚本一致）
      for ((i=0; i<${#SEQ_LIST[@]}; i++)); do
        SEQ=${SEQ_LIST[$i]}
        PRED=${PRED_LIST[$i]}
        for LR in "${LR_LIST[@]}"; do
          for DROPOUT in "${DROPOUT_LIST[@]}"; do
            for NHID in "${NHID_LIST[@]}"; do
              for WD in "${WD_LIST[@]}"; do
                run_one "Baseline" "$SEQ" "$PRED" "$LR" "$DROPOUT" "$NHID" "$WD"
              done
            done
          done
        done
      done
    done
  done
fi

# ---- 实验 2：幂律传播 ----

if [[ $EXP_ID -eq 0 || $EXP_ID -eq 2 ]]; then
  echo "==> EXP 2: PowerLaw"
  export GWN_USE_POWER=1; export GWN_USE_CHEBY=0; export GWN_DIAG_MODE=self_and_neighbor
  for DATA in "${DATA_LIST[@]}"; do for BATCH in "${BATCH_LIST[@]}"; do
    for SEQ in "${SEQ_LIST[@]}"; do for PRED in "${PRED_LIST[@]}"; do
      for LR in "${LR_LIST[@]}"; do for DROPOUT in "${DROPOUT_LIST[@]}"; do
        for NHID in "${NHID_LIST[@]}"; do for WD in "${WD_LIST[@]}"; do
          run_one "PowerLaw" "$SEQ" "$PRED" "$LR" "$DROPOUT" "$NHID" "$WD"
        done; done
      done; done
    done; done
  done; done
fi

# ---- 实验 2：PowerLaw Ablation ----
# ---- 实验 2：PowerLaw Ablation ----
# if [[ $EXP_ID -eq 0 || $EXP_ID -eq 2 ]]; then
#   echo "==> EXP 2: PowerLaw Ablation"

#   # Power Law 专用开关
#   export GWN_USE_POWER=1
#   export GWN_USE_CHEBY=0
#   export GWN_USE_MIXPROP=0

#   # ====== 实验网格 ======
#   # 阶数（2 or 3）
#   ORDER_LIST=(2 3)

#   # 幂律系数初始化策略（代码里需要根据这个开关调整实现）
#   # plain = [1,1,...]  decay = [1,0.5,0.25...]  softmax = softmax归一化
#   COEF_INIT_LIST=("plain" "decay" "softmax")

#   # 学习率
#   LR_LIST=(0.001 0.0005 0.0001)

#   # Dropout
#   DROPOUT_LIST=(0.3 0.5)

#   # Diag mode
#   DIAG_LIST=("self_and_neighbor" "neighbor")

#   for DATA in "${DATA_LIST[@]}"; do
#     for BATCH in "${BATCH_LIST[@]}"; do
#       for SEQ in "${SEQ_LIST[@]}"; do
#         for PRED in "${PRED_LIST[@]}"; do
#           for ORDER in "${ORDER_LIST[@]}"; do
#             for INIT in "${COEF_INIT_LIST[@]}"; do
#               for LR in "${LR_LIST[@]}"; do
#                 for DROPOUT in "${DROPOUT_LIST[@]}"; do
#                   for DIAG in "${DIAG_LIST[@]}"; do
#                     for NHID in "${NHID_LIST[@]}"; do
#                       for WD in "${WD_LIST[@]}"; do

#                         export GWN_POWER_ORDER=$ORDER
#                         export GWN_POWER_INIT=$INIT
#                         export GWN_DIAG_MODE=$DIAG

#                         run_one "PowerLaw_o${ORDER}_${INIT}_${DIAG}" \
#                                 "$SEQ" "$PRED" "$LR" "$DROPOUT" "$NHID" "$WD"

#                       done
#                     done
#                   done
#                 done
#               done
#             done
#           done
#         done
#       done
#     done
#   done
# fi

# ---- 实验 3：MixPropDual ----
if [[ $EXP_ID -eq 0 || $EXP_ID -eq 3 ]]; then
  echo "==> EXP 3: MixPropDual"
  export GWN_USE_MIXPROP=1
  export GWN_MIXPROP_K=${GWN_MIXPROP_K:-3}
  export GWN_ADJ_DROPOUT=${GWN_ADJ_DROPOUT:-0.1}
  export GWN_ADJ_TEMP=${GWN_ADJ_TEMP:-1.0}
  export GWN_USE_POWER=0; export GWN_USE_CHEBY=0; export GWN_DIAG_MODE=self_and_neighbor
  for DATA in "${DATA_LIST[@]}"; do for BATCH in "${BATCH_LIST[@]}"; do
    for SEQ in "${SEQ_LIST[@]}"; do for PRED in "${PRED_LIST[@]}"; do
      for LR in "${LR_LIST[@]}"; do for DROPOUT in "${DROPOUT_LIST[@]}"; do
        for NHID in "${NHID_LIST[@]}"; do for WD in "${WD_LIST[@]}"; do
          run_one "MixPropDual" "$SEQ" "$PRED" "$LR" "$DROPOUT" "$NHID" "$WD"
        done; done
      done; done
    done; done
  done; done
fi

# ---- 实验 4：Chebyshev ----
if [[ $EXP_ID -eq 0 || $EXP_ID -eq 4 ]]; then
  echo "==> EXP 4: Chebyshev"
  export GWN_USE_POWER=0; export GWN_USE_CHEBY=1
  export GWN_CHEBY_K=${GWN_CHEBY_K:-3}
  export GWN_DIAG_MODE=self_and_neighbor
  for DATA in "${DATA_LIST[@]}"; do for BATCH in "${BATCH_LIST[@]}"; do
    for SEQ in "${SEQ_LIST[@]}"; do for PRED in "${PRED_LIST[@]}"; do
      for LR in "${LR_LIST[@]}"; do for DROPOUT in "${DROPOUT_LIST[@]}"; do
        for NHID in "${NHID_LIST[@]}"; do for WD in "${WD_LIST[@]}"; do
          run_one "Chebyshev" "$SEQ" "$PRED" "$LR" "$DROPOUT" "$NHID" "$WD"
        done; done
      done; done
    done; done
  done; done
fi

# ---- 实验 5：无对角邻接 ----
if [[ $EXP_ID -eq 0 || $EXP_ID -eq 5 ]]; then
  echo "==> EXP 5: NoDiagonal"
  export GWN_USE_POWER=0; export GWN_USE_CHEBY=0; export GWN_DIAG_MODE=neighbor
  for DATA in "${DATA_LIST[@]}"; do for BATCH in "${BATCH_LIST[@]}"; do
    for SEQ in "${SEQ_LIST[@]}"; do for PRED in "${PRED_LIST[@]}"; do
      for LR in "${LR_LIST[@]}"; do for DROPOUT in "${DROPOUT_LIST[@]}"; do
        for NHID in "${NHID_LIST[@]}"; do for WD in "${WD_LIST[@]}"; do
          run_one "NoDiagonal" "$SEQ" "$PRED" "$LR" "$DROPOUT" "$NHID" "$WD"
        done; done
      done; done
    done; done
  done; done
fi

echo "✅ 实验完成（EXP_ID=$EXP_ID）。wandb 项目：$WANDB_PROJECT"
echo "📄 结果已累计写入 CSV：$RESULTS_CSV"




# DATA_LIST="data/SYNTHETIC_EASY data/SYNTHETIC_MEDIUM data/SYNTHETIC_HARD data/SYNTHETIC_VERY_HARD" BATCH_LIST="32 64 128" LR_LIST="0.001 0.0001 0.00001" EXP_ID=5 bash run_experiments.sh
# WANDB_PROJECT=GraphWaveNet-Baseline RESULTS_CSV=./results_Baseline.csv DATA_LIST="data/SYNTHETIC_EASY data/SYNTHETIC_MEDIUM data/SYNTHETIC_HARD data/SYNTHETIC_VERY_HARD" BATCH_LIST="32 64 128" LR_LIST="0.001 0.0001 0.00001" EXP_ID=1 bash run_experiments.sh
# WANDB_PROJECT=GraphWaveNet-PowerLaw RESULTS_CSV=./results_PowerLaw.csv DATA_LIST="data/SYNTHETIC_EASY data/SYNTHETIC_MEDIUM data/SYNTHETIC_HARD data/SYNTHETIC_VERY_HARD" BATCH_LIST="32 64 128" LR_LIST="0.001 0.0001 0.00001" EXP_ID=2 bash run_experiments.sh
# WANDB_PROJECT=GraphWaveNet-MixPropDual RESULTS_CSV=./results_MixPropDual.csv DATA_LIST="data/SYNTHETIC_EASY data/SYNTHETIC_MEDIUM data/SYNTHETIC_HARD data/SYNTHETIC_VERY_HARD" BATCH_LIST="32 64 128" LR_LIST="0.001 0.0001 0.00001" EXP_ID=3 bash run_experiments.sh
# WANDB_PROJECT=GraphWaveNet-Chebyshev RESULTS_CSV=./results_Chebyshev.csv DATA_LIST="data/SYNTHETIC_EASY data/SYNTHETIC_MEDIUM data/SYNTHETIC_HARD data/SYNTHETIC_VERY_HARD" BATCH_LIST="32 64 128" LR_LIST="0.001 0.0001 0.00001" EXP_ID=4 bash run_experiments.sh
# WANDB_PROJECT=GraphWaveNet-NoDiagonal RESULTS_CSV=./results_NoDiagonal.csv DATA_LIST="data/SYNTHETIC_EASY data/SYNTHETIC_MEDIUM data/SYNTHETIC_HARD data/SYNTHETIC_VERY_HARD" BATCH_LIST="32 64 128" LR_LIST="0.001 0.0001 0.00001" EXP_ID=5 bash run_experiments.sh


# WANDB_PROJECT=GraphWaveNet-PowerLaw-2 RESULTS_CSV=./results_PowerLaw-3.csv DATA_LIST="data/SYNTHETIC_HARD" BATCH_LIST="32 64 128" LR_LIST="0.001 0.0001 0.00001" EXP_ID=2 bash run_experiments.sh

# WANDB_PROJECT=GraphWaveNet-PowerLaw-2 RESULTS_CSV=./results_PowerLaw-4.csv DATA_LIST="data/SYNTHETIC_VERY_HARD" BATCH_LIST="32 64 128" LR_LIST="0.001 0.0001 0.00001" EXP_ID=2 bash run_experiments.sh


# srun -p 4090 --nodelist=aifb-websci-gpunode1 --gres=gpu:1 -t 4:00:00 --pty bash -i

# srun -p 4090 --nodelist=aifb-websci-gpunode2 --gres=gpu:1 -t 6:00:00 --pty bash -i

### nvidia-smi

### squeue -u $USER
### conda activate Energy-TSF

### cd /mnt/webscistorage/cc7738/ws_joella/EnergyTSF
# cd /mnt/webscistorage/cc7738/ws_joella/EnergyTSF/GNN/Graph-WaveNet-master-origin

# wandb login






# SYNTHETIC_EASY / SYNTHETIC_MEDIUM / SYNTHETIC_HARD / SYNTHETIC_VERY_HARD

# WANDB_PROJECT=GWN-Grid-SYNTHETIC_EASY_4 DATA=data/SYNTHETIC_EASY DEVICE=cpu EPOCHS=1 EXP_ID=4 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-SYNTHETIC_EASY_1 DATA=data/SYNTHETIC_EASY DEVICE=cuda:0 EPOCHS=500 EXP_ID=1 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-SYNTHETIC_EASY_2 DATA=data/SYNTHETIC_EASY DEVICE=cuda:0 EPOCHS=50 EXP_ID=2 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-SYNTHETIC_EASY_3 DATA=data/SYNTHETIC_EASY DEVICE=cuda:0 EPOCHS=200 EXP_ID=3 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-SYNTHETIC_EASY_4 DATA=data/SYNTHETIC_EASY DEVICE=cuda:0 EPOCHS=50 EXP_ID=4 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-SYNTHETIC_EASY_5 DATA=data/SYNTHETIC_EASY DEVICE=cuda:0 EPOCHS=50 EXP_ID=5 bash run_experiments.sh

## WANDB_PROJECT=GWN-Grid-SYNTHETIC_HARD_1 DATA=data/SYNTHETIC_HARD DEVICE=cuda:0 EPOCHS=50 EXP_ID=1 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-SYNTHETIC_HARD_2 DATA=data/SYNTHETIC_HARD DEVICE=cuda:0 EPOCHS=300 EXP_ID=2 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-SYNTHETIC_HARD_3 DATA=data/SYNTHETIC_HARD DEVICE=cuda:0 EPOCHS=50 EXP_ID=3 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-SYNTHETIC_HARD_4 DATA=data/SYNTHETIC_HARD DEVICE=cuda:0 EPOCHS=200 EXP_ID=4 bash run_experiments.sh

## WANDB_PROJECT=GWN-Grid-SYNTHETIC_HARD_5 DATA=data/SYNTHETIC_HARD DEVICE=cuda:0 EPOCHS=50 EXP_ID=5 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-SYNTHETIC_HARD_0 DATA=data/SYNTHETIC_HARD DEVICE=cuda:0 EPOCHS=50 EXP_ID=0 bash run_experiments.sh

## WANDB_PROJECT=GWN-Grid-SYNTHETIC_HARD_1_24 DATA=data/SYNTHETIC_HARD DEVICE=cuda:0 EPOCHS=100 EXP_ID=1 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-SYNTHETIC_HARD_2_24 DATA=data/SYNTHETIC_HARD DEVICE=cuda:0 EPOCHS=50 EXP_ID=2 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-SYNTHETIC_HARD_3_24 DATA=data/SYNTHETIC_HARD DEVICE=cuda:0 EPOCHS=5 EXP_ID=3 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-SYNTHETIC_HARD_4_24 DATA=data/SYNTHETIC_HARD DEVICE=cuda:0 EPOCHS=10 EXP_ID=4 bash run_experiments.sh

## WANDB_PROJECT=GWN-Grid-SYNTHETIC_HARD_5_24 DATA=data/SYNTHETIC_HARD DEVICE=cuda:0 EPOCHS=50 EXP_ID=5 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-SYNTHETIC_HARD_0_24 DATA=data/SYNTHETIC_HARD DEVICE=cuda:0 EPOCHS=50 EXP_ID=0 bash run_experiments.sh

# FRANCE / GERMANY

# WANDB_PROJECT=GWN-Grid-FRANCE_1 DATA=data/FRANCE DEVICE=cuda:0 EPOCHS=5 EXP_ID=1 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-FRANCE_2 DATA=data/FRANCE DEVICE=cuda:0 EPOCHS=5 EXP_ID=2 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-FRANCE_3 DATA=data/FRANCE DEVICE=cuda:0 EPOCHS=5 EXP_ID=3 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-FRANCE_4 DATA=data/FRANCE DEVICE=cuda:0 EPOCHS=5 EXP_ID=4 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-FRANCE_5 DATA=data/FRANCE DEVICE=cuda:0 EPOCHS=5 EXP_ID=5 bash run_experiments.sh