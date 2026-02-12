ROOT=your_path

DATA=$ROOT/data/valid.all_qwen3.parquet

OUTPUT_DIR=$ROOT/results_feb
mkdir -p $OUTPUT_DIR
cd $ROOT

# 定义三个模型路径和对应的名称
declare -a MODEL_PATHS=(
  "Qwen/Qwen3-4B-Base"
)

declare -a MODEL_NAMES=(
    "Qwen3-4B-base-ood"
)

declare -a TEMPLATES=(
    "own"
)

for i in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH="${MODEL_PATHS[$i]}"
    MODEL_NAME="${MODEL_NAMES[$i]}"
    TEMPLATE="${TEMPLATES[$i]}"
    
    echo "正在评估模型: $MODEL_NAME"
    echo "模型路径: $MODEL_PATH"
    
    for budget in 32768; do
        echo "开始生成，预算: $budget"
        python eval_scripts/generate_vllm.py \
          --model_path $MODEL_PATH \
          --input_file $DATA \
          --remove_system True \
          --output_file $OUTPUT_DIR/${MODEL_NAME}_${budget}_test.jsonl \
          --temperature 0.6 \
          --max_tokens $budget \
          --n 1 \
          --top_p 1.0 \
          --no-split-think True \
          --template $TEMPLATE > $OUTPUT_DIR/$MODEL_NAME-$budget.log
        
        echo "模型 $MODEL_NAME 评估完成"
    done
    
    echo "----------------------------------------"
done

echo "所有模型评估完成！"