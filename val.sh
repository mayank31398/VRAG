python baseline.py --eval_only --knowledge_file "data_quac/rag_format/knowledge.jsonl" --dataroot "data_quac/rag_format" --labels_file "data_quac/rag_format/train.json" --output_file "pred/RAG/pred_train.json" --model_path "runs/RAG" --checkpoint "best"

python baseline.py --eval_only --knowledge_file "data_quac/rag_format/knowledge.jsonl" --dataroot "data_quac/rag_format" --labels_file "data_quac/rag_format/val.json" --output_file "pred/RAG/pred_val.json" --model_path "runs/RAG" --checkpoint "best"

python baseline.py --eval_only --knowledge_file "data_quac/rag_format/knowledge.jsonl" --dataroot "data_quac/rag_format" --labels_file "data_quac/rag_format/test.json" --output_file "pred/RAG/pred_test.json" --model_path "runs/RAG" --checkpoint "best"