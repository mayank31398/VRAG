# ==============================================================================
# QUAC
mkdir -p "logs_quac"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_quac/RAG-5-err.log" -out "logs_quac/RAG-5-out.log" python baseline_faiss.py --params_file "configs_faiss/RAG-5.json" --dataroot "data_quac/rag_format" --model_path "runs_quac/RAG-5" --index_path "runs_quac"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_quac/VRAG-5-err.log" -out "logs_quac/VRAG-5-out.log" python baseline_faiss.py --params_file "configs_faiss/VRAG-5.json" --dataroot "data_quac/rag_format" --model_path "runs_quac/VRAG-5" --index_path "runs_quac"
# ==============================================================================


# ==============================================================================
# QUAC dialog
mkdir -p "logs_quac_dialog"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_quac_dialog/RAG-5-err.log" -out "logs_quac_dialog/RAG-5-out.log" python baseline_faiss.py --params_file "configs_faiss/RAG-5.json" --dataroot "data_quac/rag_format" --model_path "runs_quac_dialog/RAG-5" --index_path "runs_quac" --dialog

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_quac_dialog/VRAG-5-err.log" -out "logs_quac_dialog/VRAG-5-out.log" python baseline_faiss.py --params_file "configs_faiss/VRAG-5.json" --dataroot "data_quac/rag_format" --model_path "runs_quac_dialog/VRAG-5" --index_path "runs_quac" --dialog
# ==============================================================================


# ==============================================================================
# MARCO
mkdir -p "logs_marco"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_marco/RAG-5-err.log" -out "logs_marco/RAG-5-out.log" python baseline_no_index.py --params_file "configs_faiss/RAG-5.json" --dataroot "data_marco/close_format" --model_path "runs_marco/RAG-5"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_marco/VRAG-5-err.log" -out "logs_marco/VRAG-5-out.log" python baseline_no_index.py --params_file "configs_faiss/VRAG-5.json" --dataroot "data_marco/close_format" --model_path "runs_marco/VRAG-5"
# ==============================================================================