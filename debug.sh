python baseline.py --params_file "rag/configs/params.json" --knowledge_file "data_quac/debug/knowledge.jsonl" --dataroot "data_quac/debug" --model_path "runs/RAG"

python baseline.py --eval_only --params_file "rag/configs/params.json" --knowledge_file "data_quac/debug/knowledge.jsonl" --dataroot "data_quac/debug" --labels_file "data_quac/debug/val.json" --output_file "pred/RAG/pred_debug.json"