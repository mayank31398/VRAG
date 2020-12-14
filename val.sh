python baseline.py --eval_only --params_file "rag/configs/params.json" --knowledge_file "data_quac/rag_format/knowledge.jsonl" --dataroot "data_quac/rag_format" --labels_file "data_quac/rag_format/train.json" --output_file "pred/RAG/pred_train.json"

python baseline.py --eval_only --params_file "rag/configs/params.json" --knowledge_file "data_quac/rag_format/knowledge.jsonl" --dataroot "data_quac/rag_format" --labels_file "data_quac/rag_format/val.json" --output_file "pred/RAG/pred_val.json"

python baseline.py --eval_only --params_file "rag/configs/params.json" --knowledge_file "data_quac/rag_format/knowledge.jsonl" --dataroot "data_quac/rag_format" --labels_file "data_quac/rag_format/test.json" --output_file "pred/RAG/pred_test.json"