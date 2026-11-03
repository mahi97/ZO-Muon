MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

BS=${BS:-16}
EPS=${EPS:-1e-3}
TASK=${TASK:-'SST2'}
# Trainer=${Trainer:-'LowDim'}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
LR=${LR:-1e-5}
STEPS=${STEPS:-10000}
EVAL_STEPS=${EVAL_STEPS:-10000}
OPT=${OPT:-'sgd'}
MODE=${MODE:-ft}
MAX_TIME=${MAX_TIME:-0}
LOAD_MODE=${LOAD_MODE:-float16} 
SPARSITY=${SPARSITY:-1.0}
# [inverse-metric] K-FAC-shaped MeZO
KFAC_PLACEMENT=${KFAC_PLACEMENT:-A}     # A: sample with c*G^p Z A^p ; B: isotropic sample, precondition the estimate
KFAC_POWER=${KFAC_POWER:--0.5}         # -0.5: cov K-FAC^-1 ; -0.25: K-FAC^-1/2 ; 0.5: K-FAC^+1 ; 0: MeZO
KFAC_DAMPING=${KFAC_DAMPING:-1e-3}     # relative to the mean eigenvalue of each factor
KFAC_BLOCK=${KFAC_BLOCK:-512}
KFAC_CALIB_STEPS=${KFAC_CALIB_STEPS:-200}
KFAC_CALIB_MICROBS=${KFAC_CALIB_MICROBS:-16}
KFAC_FISHER=${KFAC_FISHER:-true}       # true | empirical
KFAC_REFRESH=${KFAC_REFRESH:-2000}
KFAC_CACHE_E=${KFAC_CACHE_E:-True}
STEP_INTERVAL=${STEP_INTERVAL:-50}  # [inverse-metric] unused by MeZO; was undefined -> "--step_interval --rank_r" parse error
RANK=${RANK:-2}  # [inverse-metric] unused by MeZO; was undefined

EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora"
fi
SEED=${SEED:-0}  # [inverse-metric] was hard-coded SEED=0
Tainer=kfac_mezo


LOAD_FLAG=""
if [ "$LOAD_MODE" == "bfloat16" ]; then
    LOAD_FLAG="--load_bfloat16"
elif [ "$LOAD_MODE" == "float16" ]; then
    LOAD_FLAG="--load_float16"
else
    echo "Warning: No specific load mode selected, defaulting to standard float32 or model default."
fi

case $TASK in
    CB) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        ;;
    Copa) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        TASK_ARGS="--train_as_classification False"
        ;;
    ReCoRD) 
        TASK_ARGS="--train_as_classification False"
        ;;
    DROP) 
        TASK_ARGS="--train_as_classification False"
        ;;
    SQuAD)
        TASK_ARGS="--train_as_classification False"
        ;;
    *)
        TASK_ARGS=""
        ;;
esac

TAG=$Tainer-$KFAC_PLACEMENT-p$KFAC_POWER-d$KFAC_DAMPING-$KFAC_FISHER-r$KFAC_REFRESH-$MODE-$STEPS-$BS-$LR-$EPS-$SEED


echo $TAG
echo "Task: $TASK"
echo "BS: $BS"
echo "LR: $LR"
echo "EPS: $EPS"
echo "SEED: $SEED"
echo "TRAIN/EVAL STEPS: $STEPS/$EVAL_STEPS"
echo "MODE: $MODE"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"
echo "RANK: $RANK"
echo "STEP INTERVAL: $STEP_INTERVAL"

python "$(dirname "$0")/../run.py" \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir result/$TASK-${MODEL_NAME}-$TAG --tag $TAG --seed $SEED --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
    --max_steps $STEPS \
    --trainer $Tainer \
    $LOAD_FLAG \
    --zo_optimizer $OPT \
    --adam_eps 1e-6 \
    --max_time $MAX_TIME \
    --kfac_placement $KFAC_PLACEMENT --kfac_power $KFAC_POWER --kfac_damping $KFAC_DAMPING --kfac_block $KFAC_BLOCK \
    --kfac_calib_steps $KFAC_CALIB_STEPS --kfac_calib_microbs $KFAC_CALIB_MICROBS --kfac_fisher $KFAC_FISHER \
    --kfac_refresh $KFAC_REFRESH --kfac_cache_E $KFAC_CACHE_E \
    --learning_rate $LR --zo_eps $EPS --per_device_train_batch_size $BS --lr_scheduler_type "constant" \
    --evaluation_strategy steps --save_strategy no --report_to ${REPORT_TO:-wandb} --run_name ${RUN_NAME:-$TAG} \
    --eval_steps $EVAL_STEPS --save_steps $EVAL_STEPS \
    --train_as_classification \
    --step_interval $STEP_INTERVAL \
    --rank_r $RANK \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@" 





