# ==============================================================================
# QUAC
mkdir -p "logs_quac/skip_cannot_answer"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_quac/skip_cannot_answer/RAG-5-err.log" -out "logs_quac/skip_cannot_answer/RAG-5-out.log" python baseline_faiss.py --params_file "configs_faiss/RAG-5.json" --dataroot "data_quac/rag_format" --model_path "runs_quac/skip_cannot_answer/RAG-5" --index_path "runs_quac" --skip_cannot_answer

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_quac/skip_cannot_answer/VRAG-5-err.log" -out "logs_quac/skip_cannot_answer/VRAG-5-out.log" python baseline_faiss.py --params_file "configs_faiss/VRAG-5.json" --dataroot "data_quac/rag_format" --model_path "runs_quac/skip_cannot_answer/VRAG-5" --index_path "runs_quac" --skip_cannot_answer

mkdir -p "logs_quac/weigh_cannot_answer"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_quac/weigh_cannot_answer/RAG-5-err.log" -out "logs_quac/weigh_cannot_answer/RAG-5-out.log" python baseline_faiss.py --params_file "configs_faiss/RAG-5.json" --dataroot "data_quac/rag_format" --model_path "runs_quac/weigh_cannot_answer/RAG-5" --index_path "runs_quac" --weigh_cannot_answer --weight 5

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_quac/weigh_cannot_answer/VRAG-5-err.log" -out "logs_quac/weigh_cannot_answer/VRAG-5-out.log" python baseline_faiss.py --params_file "configs_faiss/VRAG-5.json" --dataroot "data_quac/rag_format" --model_path "runs_quac/weigh_cannot_answer/VRAG-5" --index_path "runs_quac" --weigh_cannot_answer --weight 5
# ==============================================================================


# ==============================================================================
# QUAC dialog
mkdir -p "logs_quac_dialog/skip_cannot_answer"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_quac_dialog/skip_cannot_answer/RAG-5-err.log" -out "logs_quac_dialog/skip_cannot_answer/RAG-5-out.log" python baseline_faiss.py --params_file "configs_faiss/RAG-5.json" --dataroot "data_quac/rag_format" --model_path "runs_quac_dialog/skip_cannot_answer/RAG-5" --index_path "runs_quac" --dialog --skip_cannot_answer

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_quac_dialog/skip_cannot_answer/VRAG-5-err.log" -out "logs_quac_dialog/skip_cannot_answer/VRAG-5-out.log" python baseline_faiss.py --params_file "configs_faiss/VRAG-5.json" --dataroot "data_quac/rag_format" --model_path "runs_quac_dialog/skip_cannot_answer/VRAG-5" --index_path "runs_quac" --dialog --skip_cannot_answer

mkdir -p "logs_quac_dialog/weigh_cannot_answer"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_quac_dialog/weigh_cannot_answer/RAG-5-err.log" -out "logs_quac_dialog/weigh_cannot_answer/RAG-5-out.log" python baseline_faiss.py --params_file "configs_faiss/RAG-5.json" --dataroot "data_quac/rag_format" --model_path "runs_quac_dialog/weigh_cannot_answer/RAG-5" --index_path "runs_quac" --dialog --weigh_cannot_answer --weight 5

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_quac_dialog/weigh_cannot_answer/VRAG-5-err.log" -out "logs_quac_dialog/weigh_cannot_answer/VRAG-5-out.log" python baseline_faiss.py --params_file "configs_faiss/VRAG-5.json" --dataroot "data_quac/rag_format" --model_path "runs_quac_dialog/weigh_cannot_answer/VRAG-5" --index_path "runs_quac" --dialog --weigh_cannot_answer --weight 5
# ==============================================================================