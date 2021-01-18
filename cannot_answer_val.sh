# ==============================================================================
# QUAC
python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac/skip_cannot_answer/RAG-5/pred_test.json" --model_path "runs_quac/skip_cannot_answer/RAG-5" --checkpoint "best" --index_path "runs_quac" --skip_cannot_answer

python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac/skip_cannot_answer/VRAG-5/pred_test.json" --model_path "runs_quac/skip_cannot_answer/VRAG-5" --checkpoint "best" --index_path "runs_quac" --skip_cannot_answer

python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac/weigh_cannot_answer/RAG-5/pred_test.json" --model_path "runs_quac/weigh_cannot_answer/RAG-5" --checkpoint "best" --index_path "runs_quac"

python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac/weigh_cannot_answer/VRAG-5/pred_test.json" --model_path "runs_quac/weigh_cannot_answer/VRAG-5" --checkpoint "best" --index_path "runs_quac"
# ==============================================================================


# ==============================================================================
# QUAC
python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac_dialog/skip_cannot_answer/RAG-5/pred_test.json" --model_path "runs_quac_dialog/skip_cannot_answer/RAG-5" --checkpoint "best" --index_path "runs_quac" --skip_cannot_answer

python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac_dialog/skip_cannot_answer/VRAG-5/pred_test.json" --model_path "runs_quac_dialog/skip_cannot_answer/VRAG-5" --checkpoint "best" --index_path "runs_quac" --skip_cannot_answer

python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac_dialog/weigh_cannot_answer/RAG-5/pred_test.json" --model_path "runs_quac_dialog/weigh_cannot_answer/RAG-5" --checkpoint "best" --index_path "runs_quac"

python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac_dialog/weigh_cannot_answer/VRAG-5/pred_test.json" --model_path "runs_quac_dialog/weigh_cannot_answer/VRAG-5" --checkpoint "best" --index_path "runs_quac"
# ==============================================================================