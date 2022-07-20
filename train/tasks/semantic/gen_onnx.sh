eval "$(conda shell.bash hook)"
conda activate seg

# CUDA_VISIBLE_DEVICES="" python3.7 create_onnx.py -d /home/gnardari/Documents/data/ -l ../../../models/quad_bdark/ -m ../../../models/quad_bdark
# SIMBASIC DARKNET
# CUDA_VISIBLE_DEVICES="" python3.7 create_onnx.py -d /home/gnardari/Documents/data/simulated_data/ -l ../../../models/quad_erfnet/ -m ../../../models/quad_erfnet
CUDA_VISIBLE_DEVICES="" python3.7 create_onnx.py -d /home/gnardari/Documents/data/ -l ../../../models/arkansas/quad_bdark_knn/ -m ../../../models/arkansas/quad_bdark_knn
# CUDA_VISIBLE_DEVICES="" python3.7 create_onnx.py -d /home/gnardari/Documents/data/tom_labeled_files/ -l ../../../models/15tom_fast_darknet/ -m ../../../models/15tom_fast_darknet
# CUDA_VISIBLE_DEVICES="" python3.7 create_onnx.py -d /home/gnardari/Documents/data/ -l ../../../models/quad_simple_erfnet/ -m ../../../models/quad_simple_erfnet
