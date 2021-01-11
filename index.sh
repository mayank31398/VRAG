# QUAC
jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_quac/index-err.log" -out "logs_quac/index-out.log" python baseline_faiss.py --params_file "configs_faiss/RAG-5.json" --knowledge_file "data_quac/rag_format/knowledge.jsonl" --build_index --index_path "runs_quac"

# MARCO
jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_marco/index-err.log" -out "logs_marco/index-out.log" python baseline_faiss.py --params_file "configs_faiss/RAG-5.json" --knowledge_file "data_marco/rag_format/knowledge.jsonl" --build_index --index_path "runs_marco"