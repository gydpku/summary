export PYTHONFAULTHANDLER=1
source ccc_nccl.sh

# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048

# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export RANK=$((LSF_PM_XTASKID - 1))
export MASTER_ADDR=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | head -n 1)
export MASTER_PORT=54967
echo "EXP_NAME" $EXP_NAME
# echo "OUTPUT_DIR" $OUTPUT_DIR
echo "Distributed training:"
echo MASTER_ADDR $MASTER_ADDR
echo MASTER_PORT $MASTER_PORT
# echo RANK $RANK
# mkdir -p $OUTPUT_DIR

export OMP_NUM_THREADS=16
data_path=$1
model_path=$2
output_path=$3
# Train the model
# nvidia-smi
#--nnodes 1 and --nproc_per_node [num_gpus]
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 ./finetune.py --data_path "$data_path" --model_path "$model_path" --output_path "$output_path" #.py #_multi_step.py #finetune.py
#tune run --nnodes 1 --nproc_per_node 8 full_finetune_distributed --config /dccstor/obsidian_llm/yiduo/lsf/torchtune/recipes/configs/llama3/8B_full.yaml
#MKL_SERVICE_FORCE_INTEL=1
#torchrun --nnodes $WORLD_SIZE:$WORLD_SIZE \
#--nproc_per_node $NUM_GPU \
#--rdzv_id=${EXP_NAME} \
#--rdzv_backend=c10d \
#    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#    --max_restarts=1 \
#  /dccstor/obsidian_llm/yiduo/lsf/llama-finetune/src/llama_recipes/finetuning.py \
#--enable_fsdp \
#--model_name /dccstor/obsidian_llm/yiduo/h100_data/llama-3-8b \
#--use_peft \
#--peft_method lora \
#--output_dir ${EXP_DIR}

# /dccstor/obsidian_llm/yiduo/h100_data/RedWikiPedia_3_5b_95MB_chunked_393216
#> $LOG_DIR/$((LSF_PM_XTASKID - 1)).log
   # --model.model_type=${MODEL} \
  #  --model.moe_type=${MOE_TYPE} \
  #  --model.att_func=${ATT_FUNC} \
  #  --model.moe_pdrop=${MOE_PDROP} \
   # --model.sample_topk=${SAMPLETOPK} \
   # --model.gating_size=${GATING_SIZE} \
   # --model.aux_loss_type=${AUX_LOSS_TYPE} \
  #  --model.history_length=${H_LENGTH} \
 #   --model.k_att=${TOPK} \
#    --model.k_mlp=${TOPK} \ 
#pretrain_gpt.py \
 #       --moe-num-experts 8 \
  #      --moe-loss-weight 0.01 \
    #    --moe-zloss-weight 0.001 \
   #     --moe-top-k 2 \
     #   --moa \
      #  --moe-lbl-in-fp32 True \
      #  --moe-normalize-expert-weights 1 \
      #  --mlp-impl grouped \
      #  --memory-optimized-mlp \
      #  --glu \
      #  --num-layers 24 \
      #  --ffn-hidden-size 5632 \
      #  --hidden-size 2048 \
      #  --num-attention-heads 16 \
      #  --seq-length 4096 \
      #  --max-position-embeddings 4096 \
      #  --activation-function silu \
      #  --no-bias-gelu-fusion \
      #  --num-key-value-heads 8 \
      #  --kv-channels 128 \
      #  --attention-head-type multihead \
      #  --position-embedding-type rope \
      #  --normalization-function rmsnorm \
      #  --attention-dropout 0 \
      #  --hidden-dropout 0 \
      #  --micro-batch-size 4 \
      #  --global-batch-size 1024 \
      #  --train-iters 300000 \
      #  --lr-decay-iters 300000 \
      #  --lr 0.0005 \
      #  --min-lr 0.00005 \
      #  --lr-decay-style exponential \
      #  --lr-warmup-fraction 0 \
       # --override-opt_param-scheduler \
       # --clip-grad 1.0 \
       # --init-method-std 0.01 \
       # --weight-decay .1 \
       # --adam-beta2 .95 \
       # --tokenizer-type HuggingFaceTokenizer \
       # --tokenizer-path mistralai/Mistral-7B-v0.1 \
       # --make-vocab-size-divisible-by 1024 \
       # --bf16 \
       # --DDP-impl local \
       # --pipeline-model-parallel-size 3 \
       # --no-async-tensor-model-parallel-allreduce \
       # --use-flash-attn \
     #   --save-interval 2000 \
     #   --save $EXP_DIR \
     #   --load $EXP_DIR \
     #   --eval-iters 2 \
      #  --log-interval 20 \
     #   --eval-interval 200000000 \
      #  --distributed-timeout-minutes 120 \
      #  --structured-logs \
      #  --structured-logs-dir $EXP_DIR/logs \
      #  --compile-on-every-node \
       # --node-uses-local-storage \
       # --use-distributed-optimizer \
       # --num-workers 32 \
       # --fix-infiniband \
