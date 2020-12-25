# ==============================================================================
# QUAC
mkdir -p "logs_quac"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_quac/RAG-5-err.log" -out "logs_quac/RAG-5-out.log" python baseline.py --params_file "configs/RAG-5.json" --dataroot "data_quac/rag_format" --model_path "runs_quac/RAG-5" --index_path "runs_quac"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_quac/VRAG-5-err.log" -out "logs_quac/VRAG-5-out.log" python baseline.py --params_file "configs/VRAG-5.json" --dataroot "data_quac/rag_format" --model_path "runs_quac/VRAG-5" --index_path "runs_quac"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_quac/VRAG-10-err.log" -out "logs_quac/VRAG-10-out.log" python baseline.py --params_file "configs/VRAG-10.json" --dataroot "data_quac/rag_format" --model_path "runs_quac/VRAG-10" --index_path "runs_quac"
# ==============================================================================


# ==============================================================================
# MARCO
mkdir -p "logs_marco"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_marco/RAG-5-err.log" -out "logs_marco/RAG-5-out.log" python baseline.py --params_file "configs/RAG-5.json" --dataroot "data_marco/rag_format" --model_path "runs_marco/RAG-5" --index_path "runs_marco"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_marco/VRAG-5-err.log" -out "logs_marco/VRAG-5-out.log" python baseline.py --params_file "configs/VRAG-5.json" --dataroot "data_marco/rag_format" --model_path "runs_marco/VRAG-5" --index_path "runs_marco"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_marco/VRAG-10-err.log" -out "logs_marco/VRAG-10-out.log" python baseline.py --params_file "configs/VRAG-10.json" --dataroot "data_marco/rag_format" --model_path "runs_marco/VRAG-10" --index_path "runs_marco"
# ==============================================================================