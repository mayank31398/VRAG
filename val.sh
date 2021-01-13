# ==============================================================================
# QUAC
python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac/RAG-5/pred_test.json" --model_path "runs_quac/RAG-5" --checkpoint "best" --index_path "runs_quac"

python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac/VRAG-5/pred_test.json" --model_path "runs_quac/VRAG-5" --checkpoint "best" --index_path "runs_quac"

python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac/RL-5/pred_test.json" --model_path "runs_quac/RL-5" --checkpoint "best" --index_path "runs_quac"
# ==============================================================================


# ==============================================================================
# QUAC dialog
python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac_dialog/RAG-5/pred_test.json" --model_path "runs_quac_dialog/RAG-5" --checkpoint "best" --index_path "runs_quac" --dialog

python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac_dialog/VRAG-5/pred_test.json" --model_path "runs_quac_dialog/VRAG-5" --checkpoint "best" --index_path "runs_quac" --dialog

python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac_dialog/RL-5/pred_test.json" --model_path "runs_quac_dialog/RL-5" --checkpoint "best" --index_path "runs_quac" --dialog
# ==============================================================================