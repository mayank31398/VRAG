# ==============================================================================
# QUAC
python -W ignore src_faiss/scorer.py --output_file "pred_quac/skip_cannot_answer/RAG-5/pred_test.json" --score_file "pred_quac/skip_cannot_answer/RAG-5/score_test.json" --skip_cannot_answer --penalize

python -W ignore src_faiss/scorer.py --output_file "pred_quac/skip_cannot_answer/VRAG-5/pred_test.json" --score_file "pred_quac/skip_cannot_answer/VRAG-5/score_test.json" --skip_cannot_answer --penalize

python -W ignore src_faiss/scorer.py --output_file "pred_quac/DPR_fixed/pred_test.json" --score_file "pred_quac/DPR_fixed/score_test.json" --skip_cannot_answer --penalize

python -W ignore src_faiss/scorer.py --output_file "pred_quac/decoder_fine_tuning/RAG-5/pred_test.json" --score_file "pred_quac/decoder_fine_tuning/RAG-5/score_test.json" --skip_cannot_answer --penalize

python -W ignore src_faiss/scorer.py --output_file "pred_quac/decoder_fine_tuning/VRAG-5/pred_test.json" --score_file "pred_quac/decoder_fine_tuning/VRAG-5/score_test.json" --skip_cannot_answer --penalize
# # ==============================================================================


# ==============================================================================
# QUAC dialog
python -W ignore src_faiss/scorer.py --output_file "pred_quac_dialog/skip_cannot_answer/RAG-5/pred_test.json" --score_file "pred_quac_dialog/skip_cannot_answer/RAG-5/score_test.json" --skip_cannot_answer --penalize

python -W ignore src_faiss/scorer.py --output_file "pred_quac_dialog/skip_cannot_answer/VRAG-5/pred_test.json" --score_file "pred_quac_dialog/skip_cannot_answer/VRAG-5/score_test.json" --skip_cannot_answer --penalize

python -W ignore src_faiss/scorer.py --output_file "pred_quac_dialog/DPR_fixed/pred_test.json" --score_file "pred_quac_dialog/DPR_fixed/score_test.json" --skip_cannot_answer --penalize

python -W ignore src_faiss/scorer.py --output_file "pred_quac_dialog/decoder_fine_tuning/RAG-5/pred_test.json" --score_file "pred_quac_dialog/decoder_fine_tuning/RAG-5/score_test.json" --skip_cannot_answer --penalize

python -W ignore src_faiss/scorer.py --output_file "pred_quac_dialog/decoder_fine_tuning/VRAG-5/pred_test.json" --score_file "pred_quac_dialog/decoder_fine_tuning/VRAG-5/score_test.json" --skip_cannot_answer --penalize
# ==============================================================================


# ==============================================================================
# DOQA
python -W ignore src_faiss/scorer.py --output_file "pred_doqa/skip_cannot_answer/RAG-5/pred_test.json" --score_file "pred_doqa/skip_cannot_answer/RAG-5/score_test.json" --skip_cannot_answer --penalize

python -W ignore src_faiss/scorer.py --output_file "pred_doqa/skip_cannot_answer/VRAG-5/pred_test.json" --score_file "pred_doqa/skip_cannot_answer/VRAG-5/score_test.json" --skip_cannot_answer --penalize

python -W ignore src_faiss/scorer.py --output_file "pred_doqa/DPR_fixed/pred_test.json" --score_file "pred_doqa/DPR_fixed/score_test.json" --skip_cannot_answer --penalize

python -W ignore src_faiss/scorer.py --output_file "pred_doqa/decoder_fine_tuning/RAG-5/pred_test.json" --score_file "pred_doqa/decoder_fine_tuning/RAG-5/score_test.json" --skip_cannot_answer --penalize

python -W ignore src_faiss/scorer.py --output_file "pred_doqa/decoder_fine_tuning/VRAG-5/pred_test.json" --score_file "pred_doqa/decoder_fine_tuning/VRAG-5/score_test.json" --skip_cannot_answer --penalize
# ==============================================================================


# ==============================================================================
# DSTC
python -W ignore src_faiss/scorer.py --output_file "pred_dstc/RAG-5/pred_test.json" --score_file "pred_dstc/RAG-5/score_test.json" --penalize

python -W ignore src_faiss/scorer.py --output_file "pred_dstc/VRAG-5/pred_test.json" --score_file "pred_dstc/VRAG-5/score_test.json" --penalize

python -W ignore src_faiss/scorer.py --output_file "pred_dstc/DPR_fixed/pred_test.json" --score_file "pred_dstc/DPR_fixed/score_test.json" --penalize

python -W ignore src_faiss/scorer.py --output_file "pred_dstc/decoder_fine_tuning/RAG-5/pred_test.json" --score_file "pred_dstc/decoder_fine_tuning/RAG-5/score_test.json" --penalize

python -W ignore src_faiss/scorer.py --output_file "pred_dstc/decoder_fine_tuning/VRAG-5/pred_test.json" --score_file "pred_dstc/decoder_fine_tuning/VRAG-5/score_test.json" --penalize
# ==============================================================================