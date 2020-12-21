# ==============================================================================
# QUAC
# train
python baseline.py --eval_only --knowledge_file "data_quac/rag_format/knowledge.jsonl" --labels_file "data_quac/rag_format/train.json" --output_file "pred_quac/RAG-5/pred_train.json" --model_path "runs_quac/RAG-5" --checkpoint "best"

python baseline.py --eval_only --knowledge_file "data_quac/rag_format/knowledge.jsonl" --labels_file "data_quac/rag_format/train.json" --output_file "pred_quac/RAG-10/pred_train.json" --model_path "runs_quac/RAG-10" --checkpoint "best"

python baseline.py --eval_only --knowledge_file "data_quac/rag_format/knowledge.jsonl" --labels_file "data_quac/rag_format/train.json" --output_file "pred_quac/VRAG-5/pred_train.json" --model_path "runs_quac/VRAG-5" --checkpoint "best"

python baseline.py --eval_only --knowledge_file "data_quac/rag_format/knowledge.jsonl" --labels_file "data_quac/rag_format/train.json" --output_file "pred_quac/VRAG-10/pred_train.json" --model_path "runs_quac/VRAG-10" --checkpoint "best"

# val
python baseline.py --eval_only --knowledge_file "data_quac/rag_format/knowledge.jsonl" --labels_file "data_quac/rag_format/val.json" --output_file "pred_quac/RAG-5/pred_val.json" --model_path "runs_quac/RAG-5" --checkpoint "best"

python baseline.py --eval_only --knowledge_file "data_quac/rag_format/knowledge.jsonl" --labels_file "data_quac/rag_format/val.json" --output_file "pred_quac/RAG-10/pred_val.json" --model_path "runs_quac/RAG-10" --checkpoint "best"

python baseline.py --eval_only --knowledge_file "data_quac/rag_format/knowledge.jsonl" --labels_file "data_quac/rag_format/val.json" --output_file "pred_quac/VRAG-5/pred_val.json" --model_path "runs_quac/VRAG-5" --checkpoint "best"

python baseline.py --eval_only --knowledge_file "data_quac/rag_format/knowledge.jsonl" --labels_file "data_quac/rag_format/val.json" --output_file "pred_quac/VRAG-10/pred_val.json" --model_path "runs_quac/VRAG-10" --checkpoint "best"

# test
python baseline.py --eval_only --knowledge_file "data_quac/rag_format/knowledge.jsonl" --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac/RAG-5/pred_test.json" --model_path "runs_quac/RAG-5" --checkpoint "best"

python baseline.py --eval_only --knowledge_file "data_quac/rag_format/knowledge.jsonl" --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac/RAG-10/pred_test.json" --model_path "runs_quac/RAG-10" --checkpoint "best"

python baseline.py --eval_only --knowledge_file "data_quac/rag_format/knowledge.jsonl" --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac/VRAG-5/pred_test.json" --model_path "runs_quac/VRAG-5" --checkpoint "best"

python baseline.py --eval_only --knowledge_file "data_quac/rag_format/knowledge.jsonl" --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac/VRAG-10/pred_test.json" --model_path "runs_quac/VRAG-10" --checkpoint "best"
# ==============================================================================