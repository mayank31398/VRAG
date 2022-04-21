# NOTE QUAC QA dataset
# ==============================================================================
# build index (24 hours)
python baseline_faiss.py --params_file "configs_faiss/RAG-5-quac.json" --knowledge_file "data_quac/rag_format/knowledge.jsonl" --index_path "runs_quac" --build_index
# ==============================================================================


# ==============================================================================
# DPR baseline (2 days)
python baseline_faiss.py --params_file "configs_faiss/RAG-5-quac.json" --dataroot "data_quac/rag_format" --model_path "runs_quac/DPR_baseline" --index_path "runs_quac" --fix_DPR --skip_cannot_answer
# ==============================================================================


# ==============================================================================
# Vanilla
# RAG (2 days)
python baseline_faiss.py --params_file "configs_faiss/RAG-5-quac.json" --dataroot "data_quac/rag_format" --model_path "runs_quac/vanilla/RAG-5" --index_path "runs_quac" --skip_cannot_answer

# VRAG (2 days)
python baseline_faiss.py --params_file "configs_faiss/VRAG-5-quac.json" --dataroot "data_quac/rag_format" --model_path "runs_quac/vanilla/VRAG-5" --index_path "runs_quac" --skip_cannot_answer

# VRAG_magical (2 days)
python baseline_faiss.py --params_file "configs_faiss/VRAG_magical-5-quac.json" --dataroot "data_quac/rag_format" --model_path "runs_quac/vanilla/VRAG_magical-5" --index_path "runs_quac" --skip_cannot_answer
# ==============================================================================


# ==============================================================================
# Fine tune decoder
# RAG (2 days)
python baseline_faiss.py --params_file "configs_faiss/RAG-5-quac.json" --dataroot "data_quac/rag_format" --model_path "runs_quac/fine_tuned/RAG-5" --index_path "runs_quac" --fix_DPR --prior_path "runs_quac/vanilla/RAG-5/prior/best" --posterior_path "runs_quac/vanilla/RAG-5/posterior/best" --decoder_path "runs_quac/vanilla/RAG-5/decoder/best" --skip_cannot_answer

# VRAG (2 days)
python baseline_faiss.py --params_file "configs_faiss/VRAG-5-quac.json" --dataroot "data_quac/rag_format" --model_path "runs_quac/fine_tuned/VRAG-5" --index_path "runs_quac" --fix_DPR --prior_path "runs_quac/vanilla/VRAG-5/prior/best" --posterior_path "runs_quac/vanilla/VRAG-5/posterior/best" --decoder_path "runs_quac/vanilla/VRAG-5/decoder/best" --skip_cannot_answer

# VRAG_magical (2 days)
python baseline_faiss.py --params_file "configs_faiss/VRAG_magical-5-quac.json" --dataroot "data_quac/rag_format" --model_path "runs_quac/fine_tuned/VRAG_magical-5" --index_path "runs_quac" --fix_DPR --prior_path "runs_quac/vanilla/VRAG_magical-5/prior/best" --posterior_path "runs_quac/vanilla/VRAG_magical-5/posterior/best" --decoder_path "runs_quac/vanilla/VRAG_magical-5/decoder/best" --skip_cannot_answer
# ==============================================================================


# NOTE QUAC dialog dataset
# ==============================================================================
# build index (24 hours)
# already built as part of QUAC QA
# ==============================================================================


# ==============================================================================
# DPR baseline (2 days)
python baseline_faiss.py --params_file "configs_faiss/RAG-5-quac.json" --dataroot "data_quac/rag_format" --model_path "runs_quac_dialog/DPR_baseline" --index_path "runs_quac" --fix_DPR --skip_cannot_answer --dialog
# ==============================================================================


# ==============================================================================
# Vanilla
# RAG (2 days)
python baseline_faiss.py --params_file "configs_faiss/RAG-5-quac.json" --dataroot "data_quac/rag_format" --model_path "runs_quac_dialog/vanilla/RAG-5" --index_path "runs_quac" --skip_cannot_answer --dialog

# VRAG (2 days)
python baseline_faiss.py --params_file "configs_faiss/VRAG-5-quac.json" --dataroot "data_quac/rag_format" --model_path "runs_quac_dialog/vanilla/VRAG-5" --index_path "runs_quac" --skip_cannot_answer --dialog

# VRAG_magical (2 days)
python baseline_faiss.py --params_file "configs_faiss/VRAG_magical-5-quac.json" --dataroot "data_quac/rag_format" --model_path "runs_quac_dialog/vanilla/VRAG_magical-5" --index_path "runs_quac" --skip_cannot_answer --dialog
# ==============================================================================


# ==============================================================================
# Fine tune decoder
# RAG (2 days)
python baseline_faiss.py --params_file "configs_faiss/RAG-5-quac.json" --dataroot "data_quac/rag_format" --model_path "runs_quac_dialog/fine_tuned/RAG-5" --index_path "runs_quac" --fix_DPR --prior_path "runs_quac_dialog/vanilla/RAG-5/prior/best" --posterior_path "runs_quac_dialog/vanilla/RAG-5/posterior/best" --decoder_path "runs_quac_dialog/vanilla/RAG-5/decoder/best" --skip_cannot_answer --dialog

# VRAG (2 days)
python baseline_faiss.py --params_file "configs_faiss/VRAG-5-quac.json" --dataroot "data_quac/rag_format" --model_path "runs_quac_dialog/fine_tuned/VRAG-5" --index_path "runs_quac" --fix_DPR --prior_path "runs_quac_dialog/vanilla/VRAG-5/prior/best" --posterior_path "runs_quac_dialog/vanilla/VRAG-5/posterior/best" --decoder_path "runs_quac_dialog/vanilla/VRAG-5/decoder/best" --skip_cannot_answer --dialog

# VRAG_magical (2 days)
python baseline_faiss.py --params_file "configs_faiss/VRAG_magical-5.json" --dataroot "data_quac/rag_format" --model_path "runs_quac_dialog/fine_tuned/VRAG_magical-5" --index_path "runs_quac" --fix_DPR --prior_path "runs_quac_dialog/vanilla/VRAG_magical-5/prior/best" --posterior_path "runs_quac_dialog/vanilla/VRAG_magical-5/posterior/best" --decoder_path "runs_quac_dialog/vanilla/VRAG_magical-5/decoder/best" --skip_cannot_answer --dialog
# ==============================================================================


# NOTE DOQA dataset
# ==============================================================================
# build index (1 hour)
python baseline_faiss.py --params_file "configs_faiss/RAG-5-doqa.json" --knowledge_file "data_doqa/rag_format/knowledge.jsonl" --index_path "runs_doqa" --build_index
# ==============================================================================


# ==============================================================================
# DPR baseline (6 hours)
python baseline_faiss.py --params_file "configs_faiss/RAG-5-quac.json" --dataroot "data_doqa/rag_format" --model_path "runs_doqa/DPR_baseline" --index_path "runs_doqa" --dialog --skip_cannot_answer --fix_DPR
# ==============================================================================


# ==============================================================================
# Vanilla
# RAG (6 hours)
python baseline_faiss.py --params_file "configs_faiss/RAG-5-doqa.json" --dataroot "data_doqa/rag_format" --model_path "runs_doqa/vanilla/RAG-5" --index_path "runs_doqa" --dialog --skip_cannot_answer

# VRAG (6 hours)
python baseline_faiss.py --params_file "configs_faiss/VRAG-5-doqa.json" --dataroot "data_doqa/rag_format" --model_path "runs_doqa/vanilla/VRAG-5" --index_path "runs_doqa" --dialog --skip_cannot_answer

# VRAG_magical (6 hours)
python baseline_faiss.py --params_file "configs_faiss/VRAG_magical-5-doqa.json" --dataroot "data_doqa/rag_format" --model_path "runs_doqa/vanilla/VRAG_magical-5" --index_path "runs_doqa" --dialog --skip_cannot_answer
# ==============================================================================


# ==============================================================================
# Fine tune decoder
# RAG (6 hours)
python baseline_faiss.py --params_file "configs_faiss/RAG-5-quac.json" --dataroot "data_doqa/rag_format" --model_path "runs_doqa/fine_tuned/RAG-5" --index_path "runs_doqa" --fix_DPR --prior_path "runs_doqa/vanilla/RAG-5/prior/best" --posterior_path "runs_doqa/vanilla/RAG-5/posterior/best" --decoder_path "runs_doqa/vanilla/RAG-5/decoder/best" --skip_cannot_answer --dialog

# VRAG (6 hours)
python baseline_faiss.py --params_file "configs_faiss/VRAG-5-quac.json" --dataroot "data_doqa/rag_format" --model_path "runs_doqa/fine_tuned/VRAG-5" --index_path "runs_doqa" --fix_DPR --prior_path "runs_doqa/vanilla/VRAG-5/prior/best" --posterior_path "runs_doqa/vanilla/VRAG-5/posterior/best" --decoder_path "runs_doqa/vanilla/VRAG-5/decoder/best" --skip_cannot_answer --dialog

# VRAG_magical (2 days)
python baseline_faiss.py --params_file "configs_faiss/VRAG_magical-5-quac.json" --dataroot "data_doqa/rag_format" --model_path "runs_doqa/fine_tuned/VRAG_magical-5" --index_path "runs_doqa" --fix_DPR --prior_path "runs_doqa/vanilla/VRAG_magical-5/prior/best" --posterior_path "runs_doqa/vanilla/VRAG_magical-5/posterior/best" --decoder_path "runs_doqa/vanilla/VRAG_magical-5/decoder/best" --skip_cannot_answer --dialog
# ==============================================================================


# NOTE DSTC dataset
# ==============================================================================
# build index (1 hour)
python baseline_faiss.py --params_file "configs_faiss/RAG-5-dstc.json" --knowledge_file "data_dstc/rag_format/knowledge.jsonl" --index_path "runs_dstc" --build_index
# ==============================================================================


# ==============================================================================
# DPR baseline (1 day)
python baseline_faiss.py --params_file "configs_faiss/RAG-5-dstc.json" --dataroot "data_dstc/rag_format" --model_path "runs_dstc/DPR_baseline" --index_path "runs_dstc" --dialog --fix_DPR
# ==============================================================================


# ==============================================================================
# Vanilla
# RAG (1 day)
python baseline_faiss.py --params_file "configs_faiss/RAG-5-dstc.json" --dataroot "data_dstc/rag_format" --model_path "runs_dstc/vanilla/RAG-5" --index_path "runs_dstc" --dialog

# VRAG (1 day)
python baseline_faiss.py --params_file "configs_faiss/VRAG-5-dstc.json" --dataroot "data_dstc/rag_format" --model_path "runs_dstc/vanilla/VRAG-5" --index_path "runs_dstc" --dialog

# VRAG_magical (1 day)
python baseline_faiss.py --params_file "configs_faiss/VRAG_magical-5-dstc.json" --dataroot "data_dstc/rag_format" --model_path "runs_dstc/vanilla/VRAG_magical-5" --index_path "runs_dstc" --dialog
# ==============================================================================


# ==============================================================================
# Fine tune decoder
# RAG (1 day)
python baseline_faiss.py --params_file "configs_faiss/RAG-5-quac.json" --dataroot "data_dstc/rag_format" --model_path "runs_dstc/fine_tuned/RAG-5" --index_path "runs_dstc" --fix_DPR --prior_path "runs_dstc/vanilla/RAG-5/prior/best" --posterior_path "runs_dstc/vanilla/RAG-5/posterior/best" --decoder_path "runs_dstc/vanilla/RAG-5/decoder/best" --dialog

# VRAG (1 day)
python baseline_faiss.py --params_file "configs_faiss/VRAG-5-quac.json" --dataroot "data_dstc/rag_format" --model_path "runs_dstc/fine_tuned/VRAG-5" --index_path "runs_dstc" --fix_DPR --prior_path "runs_dstc/vanilla/VRAG-5/prior/best" --posterior_path "runs_dstc/vanilla/VRAG-5/posterior/best" --decoder_path "runs_dstc/vanilla/VRAG-5/decoder/best" --dialog

# VRAG_magical (2 days)
python baseline_faiss.py --params_file "configs_faiss/VRAG_magical-5-quac.json" --dataroot "data_dstc/rag_format" --model_path "runs_dstc/fine_tuned/VRAG_magical-5" --index_path "runs_dstc" --fix_DPR --prior_path "runs_dstc/vanilla/VRAG_magical-5/prior/best" --posterior_path "runs_dstc/vanilla/VRAG_magical-5/posterior/best" --decoder_path "runs_dstc/vanilla/VRAG_magical-5/decoder/best" --skip_cannot_answer --dialog
# ==============================================================================