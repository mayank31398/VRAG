mkdir -p "logs"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs/RAG-err.log" -out "logs/RAG-out.log" python baseline.py --params_file "rag/configs/RAG.json" --knowledge_file "data_quac/rag_format/knowledge.jsonl" --dataroot "data_quac/rag_format" --model_path "runs/RAG"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs/VRAG-err.log" -out "logs/VRAG-out.log" python baseline.py --params_file "rag/configs/VRAG.json" --knowledge_file "data_quac/rag_format/knowledge.jsonl" --dataroot "data_quac/rag_format" --model_path "runs/VRAG"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs/VRAG_union-err.log" -out "logs/VRAG_union-out.log" python baseline.py --params_file "rag/configs/VRAG_union.json" --knowledge_file "data_quac/rag_format/knowledge.jsonl" --dataroot "data_quac/rag_format" --model_path "runs/VRAG_union"