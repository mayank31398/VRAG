python baseline.py --params_file "configs/VRAG-5.json" --knowledge_file "data_quac/debug/knowledge.jsonl" --build_index --index_path "tmp"

python baseline.py --params_file "configs/VRAG-5.json" --dataroot "data_quac/debug" --model_path "tmp" --index_path "tmp"