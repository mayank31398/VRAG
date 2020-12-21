# ==============================================================================
# Prepare datasets for RAG
python src/data/create_dataset_quac.py
python src/data/augment_data_marco.py
python src/data/create_dataset_marco.py
# ==============================================================================


# ==============================================================================
# QUAC
mkdir -p "logs_quac"

# build index
python baseline.py --params_file "configs/RAG-5.json" --knowledge_file "data_quac/rag_format/knowledge.jsonl" --build_index --index_path "runs_quac"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_quac/RAG-5-err.log" -out "logs_quac/RAG-5-out.log" python baseline.py --params_file "configs/RAG-5.json" --dataroot "data_quac/rag_format" --model_path "runs_quac/RAG-5"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_quac/RAG-10-err.log" -out "logs_quac/RAG-10-out.log" python baseline.py --params_file "configs/RAG-10.json" --dataroot "data_quac/rag_format" --model_path "runs_quac/RAG-10"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_quac/VRAG-5-err.log" -out "logs_quac/VRAG-5-out.log" python baseline.py --params_file "configs/VRAG-5.json" --dataroot "data_quac/rag_format" --model_path "runs_quac/VRAG-5"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_quac/VRAG-10-err.log" -out "logs_quac/VRAG-10-out.log" python baseline.py --params_file "configs/VRAG-10.json" --dataroot "data_quac/rag_format" --model_path "runs_quac/VRAG-10"
# ==============================================================================


# ==============================================================================
# MARCO
mkdir -p "logs_marco"

# build index
python baseline.py --params_file "configs/RAG-5.json" --knowledge_file "data_marco/rag_format/knowledge.jsonl" --build_index --index_path "runs_marco"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_marco/RAG-5-err.log" -out "logs_marco/RAG-5-out.log" python baseline.py --params_file "configs/RAG-5.json" --dataroot "data_marco/rag_format" --model_path "runs_marco/RAG-5"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_marco/RAG-10-err.log" -out "logs_marco/RAG-10-out.log" python baseline.py --params_file "configs/RAG-10.json" --dataroot "data_marco/rag_format" --model_path "runs_marco/RAG-10"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_marco/VRAG-5-err.log" -out "logs_marco/VRAG-5-out.log" python baseline.py --params_file "configs/VRAG-5.json" --dataroot "data_marco/rag_format" --model_path "runs_marco/VRAG-5"

jbsub -q x86_7d -cores 1x1+1 -require v100 -mem 100G -err "logs_marco/VRAG-10-err.log" -out "logs_marco/VRAG-10-out.log" python baseline.py --params_file "configs/VRAG-10.json" --dataroot "data_marco/rag_format" --model_path "runs_marco/VRAG-10"
# ==============================================================================