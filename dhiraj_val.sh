# NOTE change checkpont name

# ==============================================================================
# QUAC
python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac/DPR_fixed/pred_test.json" --model_path "runs_quac/DPR_fixed" --checkpoint "best" --index_path "runs_quac"
# ==============================================================================


# ==============================================================================
# QUAC dialog
python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac_dialog/DPR_fixed/pred_test.json" --model_path "runs_quac_dialog/DPR_fixed" --checkpoint "best" --index_path "runs_quac" --dialog
# ==============================================================================