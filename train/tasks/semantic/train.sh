eval "$(conda shell.bash hook)"
conda activate seg

python3.7 ./train.py -d /home/gnardari/Documents/data/ -ac ./config/arch/quad/quad_bdarknet.yaml -dc ./config/labels/quad_binary.yaml -l ../../../models/arkansas/quad_bdark_knn/ -p ../../../models/sim/sim_quad_bdark
# python3.7 ./train.py -d /home/gnardari/Documents/data/ -ac ./config/arch/quad/quad_erfnet.yaml -dc ./config/labels/quad_binary.yaml -l ../../../models/quad_erfnet/ -p ../../../models/quad_sim_erfnet
# python3.7 ./train.py -d /home/gnardari/Documents/data/800_data/ -ac ./config/arch/quad/quad_bdarknet_800m.yaml -dc ./config/labels/quad_binary.yaml -l ../../../models/800m/quad_bdark/ -p ../../../models/sim/sim_quad_bdark
# python3.7 ./train.py -d /home/gnardari/Documents/data/800_data/ -ac ./config/arch/quad/quad_erfnet.yaml -dc ./config/labels/quad_binary.yaml -l ../../../models/800m/quad_erfnet/ -p ../../../models/sim/quad_sim_erfnet
# PRETRAINED WITH SIM
# python3.7 ./train.py -d /home/gnardari/Documents/data/ -ac ./config/arch/quad_squeezesegv2.yaml -dc ./config/labels/quad_binary.yaml -l ../../../models/quad_squeeze/ -p ../../../models/sim_quad_squeeze
