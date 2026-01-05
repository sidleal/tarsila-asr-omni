export OUTPUT_DIR="/home/jovyan/omnilingual-asr/output"
python -m workflows.recipes.wav2vec2.asr $OUTPUT_DIR --config-file workflows/recipes/wav2vec2/asr/configs/llm-finetune-tarsila.yaml