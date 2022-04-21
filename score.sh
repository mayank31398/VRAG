# NOTE QUAC QA dataset
# ==============================================================================
# DPR baseline
python -W ignore src_faiss/scorer.py --output_file "pred_quac/DPR_baseline/pred_test.json" --score_file "pred_quac/DPR_baseline/score_test.json" --skip_cannot_answer
# ==============================================================================


# ==============================================================================
# Vanilla
# RAG
python -W ignore src_faiss/scorer.py --output_file "pred_quac/vanilla/RAG-5/pred_test.json" --score_file "pred_quac/vanilla/RAG-5/score_test.json" --skip_cannot_answer

# VRAG
python -W ignore src_faiss/scorer.py --output_file "pred_quac/vanilla/VRAG-5/pred_test.json" --score_file "pred_quac/vanilla/VRAG-5/score_test.json" --skip_cannot_answer

# VRAG_magical
python -W ignore src_faiss/scorer.py --output_file "pred_quac/vanilla/VRAG_magical-5/pred_test.json" --score_file "pred_quac/vanilla/VRAG_magical-5/score_test.json" --skip_cannot_answer
# ==============================================================================


# ==============================================================================
# Fine tune decoder
# RAG
python -W ignore src_faiss/scorer.py --output_file "pred_quac/fine_tuned/RAG-5/pred_test.json" --score_file "pred_quac/fine_tuned/RAG-5/score_test.json" --skip_cannot_answer

# VRAG
python -W ignore src_faiss/scorer.py --output_file "pred_quac/fine_tuned/VRAG-5/pred_test.json" --score_file "pred_quac/fine_tuned/VRAG-5/score_test.json" --skip_cannot_answer

# VRAG_magical
python -W ignore src_faiss/scorer.py --output_file "pred_quac/fine_tuned/VRAG_magical-5/pred_test.json" --score_file "pred_quac/fine_tuned/VRAG_magical-5/score_test.json" --skip_cannot_answer
# ==============================================================================


# NOTE QUAC dialog dataset
# ==============================================================================
# DPR baseline
python -W ignore src_faiss/scorer.py --output_file "pred_quac_dialog/DPR_baseline/pred_test.json" --score_file "pred_quac_dialog/DPR_baseline/score_test.json" --skip_cannot_answer
# ==============================================================================


# ==============================================================================
# Vanilla
# RAG
python -W ignore src_faiss/scorer.py --output_file "pred_quac_dialog/vanilla/RAG-5/pred_test.json" --score_file "pred_quac_dialog/vanilla/RAG-5/score_test.json" --skip_cannot_answer

# VRAG
python -W ignore src_faiss/scorer.py --output_file "pred_quac_dialog/vanilla/VRAG-5/pred_test.json" --score_file "pred_quac_dialog/vanilla/VRAG-5/score_test.json" --skip_cannot_answer

# VRAG_magical
python -W ignore src_faiss/scorer.py --output_file "pred_quac_dialog/vanilla/VRAG_magical-5/pred_test.json" --score_file "pred_quac_dialog/vanilla/VRAG_magical-5/score_test.json" --skip_cannot_answer
# ==============================================================================


# ==============================================================================
# Fine tune decoder
# RAG
python -W ignore src_faiss/scorer.py --output_file "pred_quac_dialog/fine_tuned/RAG-5/pred_test.json" --score_file "pred_quac_dialog/fine_tuned/RAG-5/score_test.json" --skip_cannot_answer

# VRAG
python -W ignore src_faiss/scorer.py --output_file "pred_quac_dialog/fine_tuned/VRAG-5/pred_test.json" --score_file "pred_quac_dialog/fine_tuned/VRAG-5/score_test.json" --skip_cannot_answer

# VRAG_magical
python -W ignore src_faiss/scorer.py --output_file "pred_quac_dialog/fine_tuned/VRAG_magical-5/pred_test.json" --score_file "pred_quac_dialog/fine_tuned/VRAG_magical-5/score_test.json" --skip_cannot_answer
# ==============================================================================


# NOTE DOQA dataset
# ==============================================================================
# DPR baseline
python -W ignore src_faiss/scorer.py --output_file "pred_doqa/DPR_baseline/pred_test.json" --score_file "pred_doqa/DPR_baseline/score_test.json" --skip_cannot_answer
# ==============================================================================


# ==============================================================================
# Vanilla
# RAG
python -W ignore src_faiss/scorer.py --output_file "pred_doqa/vanilla/RAG-5/pred_test.json" --score_file "pred_doqa/vanilla/RAG-5/score_test.json" --skip_cannot_answer

# VRAG
python -W ignore src_faiss/scorer.py --output_file "pred_doqa/vanilla/VRAG-5/pred_test.json" --score_file "pred_doqa/vanilla/VRAG-5/score_test.json" --skip_cannot_answer

# VRAG_magical
python -W ignore src_faiss/scorer.py --output_file "pred_doqa/vanilla/VRAG_magical-5/pred_test.json" --score_file "pred_doqa/vanilla/VRAG_magical-5/score_test.json" --skip_cannot_answer
# ==============================================================================


# ==============================================================================
# Fine tune decoder
# RAG
python -W ignore src_faiss/scorer.py --output_file "pred_doqa/fine_tuned/RAG-5/pred_test.json" --score_file "pred_doqa/fine_tuned/RAG-5/score_test.json" --skip_cannot_answer

# VRAG
python -W ignore src_faiss/scorer.py --output_file "pred_doqa/fine_tuned/VRAG-5/pred_test.json" --score_file "pred_doqa/fine_tuned/VRAG-5/score_test.json" --skip_cannot_answer

# VRAG_magical
python -W ignore src_faiss/scorer.py --output_file "pred_doqa/fine_tuned/VRAG_magical-5/pred_test.json" --score_file "pred_doqa/fine_tuned/VRAG_magical-5/score_test.json" --skip_cannot_answer
# ==============================================================================


# NOTE DSTC dataset
# ==============================================================================
# DPR baseline
python -W ignore src_faiss/scorer.py --output_file "pred_dstc/DPR_baseline/pred_test.json" --score_file "pred_dstc/DPR_baseline/score_test.json" --skip_cannot_answer
# ==============================================================================


# ==============================================================================
# Vanilla
# RAG
python -W ignore src_faiss/scorer.py --output_file "pred_dstc/vanilla/RAG-5/pred_test.json" --score_file "pred_dstc/vanilla/RAG-5/score_test.json" --skip_cannot_answer

# VRAG
python -W ignore src_faiss/scorer.py --output_file "pred_dstc/vanilla/VRAG-5/pred_test.json" --score_file "pred_dstc/vanilla/VRAG-5/score_test.json" --skip_cannot_answer

# VRAG_magical
python -W ignore src_faiss/scorer.py --output_file "pred_dstc/vanilla/VRAG_magical-5/pred_test.json" --score_file "pred_dstc/vanilla/VRAG_magical-5/score_test.json" --skip_cannot_answer
# ==============================================================================


# ==============================================================================
# Fine tune decoder
# RAG
python -W ignore src_faiss/scorer.py --output_file "pred_dstc/fine_tuned/RAG-5/pred_test.json" --score_file "pred_dstc/fine_tuned/RAG-5/score_test.json" --skip_cannot_answer

# VRAG
python -W ignore src_faiss/scorer.py --output_file "pred_dstc/fine_tuned/VRAG-5/pred_test.json" --score_file "pred_dstc/fine_tuned/VRAG-5/score_test.json" --skip_cannot_answer

# VRAG_magical
python -W ignore src_faiss/scorer.py --output_file "pred_dstc/fine_tuned/VRAG_magical-5/pred_test.json" --score_file "pred_dstc/fine_tuned/VRAG_magical-5/score_test.json" --skip_cannot_answer
# ==============================================================================