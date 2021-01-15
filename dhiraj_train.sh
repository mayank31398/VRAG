# NOTE change stopping criteria to 10 in config file used in thre following

# ==============================================================================
# QUAC
mkdir -p "logs_quac/DPR_fixed"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_quac/DPR_fixed/err.log" -out "logs_quac/DPR_fixed/out.log" python baseline_faiss.py --params_file "configs_faiss/RAG-5.json" --dataroot "data_quac/rag_format" --model_path "runs_quac/DPR_fixed" --index_path "runs_quac" --fix_DPR
# ==============================================================================


# ==============================================================================
# QUAC dialog
mkdir -p "logs_quac_dialog/DPR_fixed"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_quac_dialog/DPR_fixed/err.log" -out "logs_quac_dialog/DPR_fixed/out.log" python baseline_faiss.py --params_file "configs_faiss/RAG-5.json" --dataroot "data_quac/rag_format" --model_path "runs_quac_dialog/DPR_fixed" --index_path "runs_quac" --dialog --fix_DPR
# ==============================================================================