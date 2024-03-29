# find all configs in configs/
config=activitynet
# set your gpu id
gpus=5
# number of gpus
gpun=1
# please modify it with different value (e.g., 127.0.0.2, 29502) when you run multi mmn task on the same machine
master_addr=127.0.0.3
master_port=29950

# ------------------------ need not change -----------------------------------
config_file=configs/$config\.yaml
output_dir=outputs/$config

#CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch \
#--nproc_per_node=$gpun --master_addr $master_addr --master_port $master_port
python  train_net.py --config-file $config_file OUTPUT_DIR $output_dir 
