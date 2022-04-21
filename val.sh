# NOTE QUAC QA dataset
# ==============================================================================
# DPR baseline
python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac/DPR_baseline/pred_test.json" --model_path "runs_quac/DPR_baseline" --checkpoint "checkpoint-129710" --index_path "runs_quac" --skip_cannot_answer
# ==============================================================================


# ==============================================================================
# Vanilla
# RAG
python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac/vanilla/RAG-5/pred_test.json" --model_path "runs_quac/vanilla/RAG-5" --checkpoint "best" --index_path "runs_quac" --skip_cannot_answer

# VRAG
python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac/vanilla/VRAG-5/pred_test.json" --model_path "runs_quac/vanilla/VRAG-5" --checkpoint "best" --index_path "runs_quac" --skip_cannot_answer

# VRAG_magical
python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac/vanilla/VRAG_magical-5/pred_test.json" --model_path "runs_quac/vanilla/VRAG_magical-5" --checkpoint "best" --index_path "runs_quac" --skip_cannot_answer
# ==============================================================================


# ==============================================================================
# Fine tune decoder
# RAG
python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac/fine_tuned/RAG-5/pred_test.json" --model_path "runs_quac/fine_tuned/RAG-5" --checkpoint "checkpoint-129710" --index_path "runs_quac" --skip_cannot_answer

# VRAG
python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac/fine_tuned/VRAG-5/pred_test.json" --model_path "runs_quac/fine_tuned/VRAG-5" --checkpoint "checkpoint-129710" --index_path "runs_quac" --skip_cannot_answer

# VRAG_magical
python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac/fine_tuned/VRAG_magical-5/pred_test.json" --model_path "runs_quac/fine_tuned/VRAG_magical-5" --checkpoint "checkpoint-129710" --index_path "runs_quac" --skip_cannot_answer
# ==============================================================================


# NOTE QUAC dialog dataset
# ==============================================================================
# DPR baseline
python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac_dialog/DPR_baseline/pred_test.json" --model_path "runs_quac_dialog/DPR_baseline" --checkpoint "checkpoint-129710" --index_path "runs_quac" --skip_cannot_answer --dialog
# ==============================================================================


# ==============================================================================
# Vanilla
# RAG
python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac_dialog/vanilla/RAG-5/pred_test.json" --model_path "runs_quac_dialog/vanilla/RAG-5" --checkpoint "best" --index_path "runs_quac" --skip_cannot_answer --dialog

# VRAG
python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac_dialog/vanilla/VRAG-5/pred_test.json" --model_path "runs_quac_dialog/vanilla/VRAG-5" --checkpoint "best" --index_path "runs_quac" --skip_cannot_answer --dialog

# VRAG_magical
python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac_dialog/vanilla/VRAG_magical-5/pred_test.json" --model_path "runs_quac_dialog/vanilla/VRAG_magical-5" --checkpoint "best" --index_path "runs_quac" --skip_cannot_answer --dialog
# ==============================================================================


# ==============================================================================
# Fine tune decoder
# RAG
python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac_dialog/fine_tuned/RAG-5/pred_test.json" --model_path "runs_quac_dialog/fine_tuned/RAG-5" --checkpoint "checkpoint-129710" --index_path "runs_quac" --skip_cannot_answer --dialog

# VRAG
python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac_dialog/fine_tuned/VRAG-5/pred_test.json" --model_path "runs_quac_dialog/fine_tuned/VRAG-5" --checkpoint "checkpoint-129710" --index_path "runs_quac" --skip_cannot_answer --dialog

# VRAG_magical
python baseline_faiss.py --eval_only --labels_file "data_quac/rag_format/test.json" --output_file "pred_quac_dialog/fine_tuned/VRAG_magical-5/pred_test.json" --model_path "runs_quac_dialog/fine_tuned/VRAG_magical-5" --checkpoint "checkpoint-129710" --index_path "runs_quac" --skip_cannot_answer --dialog
# ==============================================================================


# NOTE DOQA dataset
# ==============================================================================
# DPR baseline
python baseline_faiss.py --eval_only --labels_file "data_doqa/rag_format/test.json" --output_file "pred_doqa/DPR_baseline/pred_test.json" --model_path "runs_doqa/DPR_baseline" --checkpoint "checkpoint-129710" --index_path "runs_doqa" --skip_cannot_answer --dialog
# ==============================================================================


# ==============================================================================
# Vanilla
# RAG
python baseline_faiss.py --eval_only --labels_file "data_doqa/rag_format/test.json" --output_file "pred_doqa/vanilla/RAG-5/pred_test.json" --model_path "runs_doqa/vanilla/RAG-5" --checkpoint "best" --index_path "runs_doqa" --skip_cannot_answer --dialog

# VRAG
python baseline_faiss.py --eval_only --labels_file "data_doqa/rag_format/test.json" --output_file "pred_doqa/vanilla/VRAG-5/pred_test.json" --model_path "runs_doqa/vanilla/VRAG-5" --checkpoint "best" --index_path "runs_doqa" --skip_cannot_answer --dialog

# VRAG_magical
python baseline_faiss.py --eval_only --labels_file "data_doqa/rag_format/test.json" --output_file "pred_doqa/vanilla/VRAG_magical-5/pred_test.json" --model_path "runs_doqa/vanilla/VRAG_magical-5" --checkpoint "best" --index_path "runs_doqa" --skip_cannot_answer --dialog
# ==============================================================================


# ==============================================================================
# Fine tune decoder
# RAG
python baseline_faiss.py --eval_only --labels_file "data_doqa/rag_format/test.json" --output_file "pred_doqa/fine_tuned/RAG-5/pred_test.json" --model_path "runs_doqa/fine_tuned/RAG-5" --checkpoint "checkpoint-129710" --index_path "runs_doqa" --skip_cannot_answer --dialog

# VRAG
python baseline_faiss.py --eval_only --labels_file "data_doqa/rag_format/test.json" --output_file "pred_doqa/fine_tuned/VRAG-5/pred_test.json" --model_path "runs_doqa/fine_tuned/VRAG-5" --checkpoint "checkpoint-129710" --index_path "runs_doqa" --skip_cannot_answer --dialog

# VRAG_magical
python baseline_faiss.py --eval_only --labels_file "data_doqa/rag_format/test.json" --output_file "pred_doqa/fine_tuned/VRAG_magical-5/pred_test.json" --model_path "runs_doqa/fine_tuned/VRAG_magical-5" --checkpoint "checkpoint-129710" --index_path "runs_doqa" --skip_cannot_answer --dialog
# ==============================================================================


# NOTE DSTC dataset
# ==============================================================================
# DPR baseline
python baseline_faiss.py --eval_only --labels_file "data_dstc/rag_format/test.json" --output_file "pred_dstc/DPR_baseline/pred_test.json" --model_path "runs_dstc/DPR_baseline" --checkpoint "checkpoint-129710" --index_path "runs_dstc" --skip_cannot_answer --dialog
# ==============================================================================


# ==============================================================================
# Vanilla
# RAG
python baseline_faiss.py --eval_only --labels_file "data_dstc/rag_format/test.json" --output_file "pred_dstc/vanilla/RAG-5/pred_test.json" --model_path "runs_dstc/vanilla/RAG-5" --checkpoint "best" --index_path "runs_dstc" --skip_cannot_answer --dialog

# VRAG
python baseline_faiss.py --eval_only --labels_file "data_dstc/rag_format/test.json" --output_file "pred_dstc/vanilla/VRAG-5/pred_test.json" --model_path "runs_dstc/vanilla/VRAG-5" --checkpoint "best" --index_path "runs_dstc" --skip_cannot_answer --dialog

# VRAG_magical
python baseline_faiss.py --eval_only --labels_file "data_dstc/rag_format/test.json" --output_file "pred_dstc/vanilla/VRAG_magical-5/pred_test.json" --model_path "runs_dstc/vanilla/VRAG_magical-5" --checkpoint "best" --index_path "runs_dstc" --skip_cannot_answer --dialog
# ==============================================================================


# ==============================================================================
# Fine tune decoder
# RAG
python baseline_faiss.py --eval_only --labels_file "data_dstc/rag_format/test.json" --output_file "pred_dstc/fine_tuned/RAG-5/pred_test.json" --model_path "runs_dstc/fine_tuned/RAG-5" --checkpoint "checkpoint-129710" --index_path "runs_dstc" --skip_cannot_answer --dialog

# VRAG
python baseline_faiss.py --eval_only --labels_file "data_dstc/rag_format/test.json" --output_file "pred_dstc/fine_tuned/VRAG-5/pred_test.json" --model_path "runs_dstc/fine_tuned/VRAG-5" --checkpoint "checkpoint-129710" --index_path "runs_dstc" --skip_cannot_answer --dialog

# VRAG_magical
python baseline_faiss.py --eval_only --labels_file "data_dstc/rag_format/test.json" --output_file "pred_dstc/fine_tuned/VRAG_magical-5/pred_test.json" --model_path "runs_dstc/fine_tuned/VRAG_magical-5" --checkpoint "checkpoint-129710" --index_path "runs_dstc" --skip_cannot_answer --dialog
# ==============================================================================