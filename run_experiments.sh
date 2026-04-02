#!/bin/bash

# ==============================================================================
# Model Experiment Orchestrator for FactoryBench Level 1
# ==============================================================================
# 
# Usage: ./run_experiments.sh [questions_per_template]
# Example: ./run_experiments.sh 1
# ==============================================================================

# List of models to evaluate
MODELS=(
    "gpt-4o-mini"
    "student-gpt-4.1"
)

# questions per template
QUESTIONS_PER_TEMPLATE=${1:-1}

for MODEL in "${MODELS[@]}"
do
    echo "============================================================"
    echo "STARTING EVALUATION FOR MODEL: $MODEL"
    echo "Questions per template: $QUESTIONS_PER_TEMPLATE"
    echo "============================================================"
    
    python src/pipeline/run_level1_pipeline.py \
        -t $QUESTIONS_PER_TEMPLATE \
        --test-mode \
        --model "$MODEL" || {
            echo "Error running pipeline for $MODEL"
            continue
        }
        
    echo ""
    echo "Completed evaluation for $MODEL"
    echo "============================================================"
    echo ""
done

echo "All experiments complete!"
