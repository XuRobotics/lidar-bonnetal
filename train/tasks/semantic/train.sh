eval "$(conda shell.bash hook)"
conda activate seg
python3.7 ./train.py -d /home/gnardari/Documents/data/ -ac ./config/arch/quad_darknet21.yaml -dc ./config/labels/quad_binary.yaml -l ../../../quad_log/ -p ../../../quad_darknet21/