# ==============================================================================
# QUAC faiss
mkdir -p "logs_quac"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_quac/RAG-5-err.log" -out "logs_quac/RAG-5-out.log" python baseline_faiss.py --params_file "configs_faiss/RAG-5.json" --dataroot "data_quac/rag_format" --model_path "runs_quac/RAG-5" --index_path "runs_quac"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_quac/VRAG-5-err.log" -out "logs_quac/VRAG-5-out.log" python baseline_faiss.py --params_file "configs_faiss/VRAG-5.json" --dataroot "data_quac/rag_format" --model_path "runs_quac/VRAG-5" --index_path "runs_quac"
# ==============================================================================


# ==============================================================================
# QUAC annoy
mkdir -p "logs_quac"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_quac/annoy-err.log" -out "logs_quac/annoy-out.log" python baseline_annoy.py --params_file "configs_annoy/RAG-5.json" --index_path "runs_quac_annoy" --build_index --knowledge_file "data_quac/rag_format/knowledge.jsonl"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_quac/RAG_annoy-5-err.log" -out "logs_quac/RAG_annoy-5-out.log" python baseline_annoy.py --params_file "configs_annoy/RAG-5.json" --dataroot "data_quac/rag_format" --model_path "runs_quac_annoy/RAG-5" --index_path "runs_quac_annoy" --knowledge_file "data_quac/rag_format/knowledge.jsonl"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_quac/VRAG_annoy-5-err.log" -out "logs_quac/VRAG_annoy-5-out.log" python baseline_annoy.py --params_file "configs_annoy/VRAG-5.json" --dataroot "data_quac/rag_format" --model_path "runs_quac_annoy/VRAG-5" --index_path "runs_quac_annoy" --knowledge_file "data_quac/rag_format/knowledge.jsonl"
# # ==============================================================================