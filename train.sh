mkdir -p "logs"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs/RAG-5-err.log" -out "logs/RAG-5-out.log" python baseline_quac.py --params_file "baseline_quac/configs/RAG-5.json" --knowledge_file "data_quac/rag_format/knowledge.jsonl" --dataroot "data_quac/rag_format" --model_path "runs/RAG-5"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs/VRAG-5-err.log" -out "logs/VRAG-5-out.log" python baseline_quac.py --params_file "baseline_quac/configs/VRAG-5.json" --knowledge_file "data_quac/rag_format/knowledge.jsonl" --dataroot "data_quac/rag_format" --model_path "runs/VRAG-5"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs/RAG-10-err.log" -out "logs/RAG-10-out.log" python baseline_quac.py --params_file "baseline_quac/configs/RAG-10.json" --knowledge_file "data_quac/rag_format/knowledge.jsonl" --dataroot "data_quac/rag_format" --model_path "runs/RAG-10"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs/VRAG-10-err.log" -out "logs/VRAG-10-out.log" python baseline_quac.py --params_file "baseline_quac/configs/VRAG-10.json" --knowledge_file "data_quac/rag_format/knowledge.jsonl" --dataroot "data_quac/rag_format" --model_path "runs/VRAG-10"