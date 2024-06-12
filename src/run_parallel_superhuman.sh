pip3 uninstall -y timm &&
pip3 install timm==0.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple  &&
sed -i 's/from torch._six import container_abcs/import collections.abc as container_abcs/g' /usr/local/lib/python3.8/dist-packages/timm/models/layers/helpers.py &&
export NCCL_P2P_DISABLE=1 &&
echo "Starting task on GPU 0 1"
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=46667 /h3cstore_ns/hyshi/EM_mamba_new/EM_mamba_seg/main_finetune.py --batch_size=6 --crop_size=18,160,160 \
--epochs=800 --output_dir=/h3cstore_ns/hyshi/EM_mamba_new/result/superhuman_ac3_lr5_b6_18_160_160 \
--visual_dir=/h3cstore_ns/hyshi/EM_mamba_new/result/superhuman_ac3_lr5_b6_18_160_160/visual \
--log_dir=/h3cstore_ns/hyshi/EM_mamba_new/result/superhuman_ac3_lr5_b6_18_160_160/tensorboard_log \
--warmup_epochs=0 --blr=1e-5 &
echo "Starting task on GPU 2 3"
CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=46668 /h3cstore_ns/hyshi/EM_mamba_new/EM_mamba_seg/main_finetune.py --batch_size=6 --crop_size=16,160,160 \
--epochs=800 --output_dir=/h3cstore_ns/hyshi/EM_mamba_new/result/superhuman_ac3_lr5_b6_16_160_160 \
--visual_dir=/h3cstore_ns/hyshi/EM_mamba_new/result/superhuman_ac3_lr5_b6_16_160_160/visual \
--log_dir=/h3cstore_ns/hyshi/EM_mamba_new/result/superhuman_ac3_lr5_b6_16_160_160/tensorboard_log \
--warmup_epochs=0 --blr=1e-5 &
echo "Starting task on GPU 4 5"
CUDA_VISIBLE_DEVICES=4,5 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=46669 /h3cstore_ns/hyshi/EM_mamba_new/EM_mamba_seg/main_finetune.py --batch_size=2 --crop_size=18,160,160 \
--epochs=800 --output_dir=/h3cstore_ns/hyshi/EM_mamba_new/result/superhuman_ac3_lr5_b2_18_160_160 \
--visual_dir=/h3cstore_ns/hyshi/EM_mamba_new/result/superhuman_ac3_lr5_b2_18_160_160/visual \
--log_dir=/h3cstore_ns/hyshi/EM_mamba_new/result/superhuman_ac3_lr5_b2_18_160_160/tensorboard_log \
--warmup_epochs=0 --blr=1e-5 &
echo "Starting task on GPU 6 7"
CUDA_VISIBLE_DEVICES=6,7 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=46670 /h3cstore_ns/hyshi/EM_mamba_new/EM_mamba_seg/main_finetune.py --batch_size=2 --crop_size=16,160,160 \
--epochs=800 --output_dir=/h3cstore_ns/hyshi/EM_mamba_new/result/superhuman_ac3_lr5_b2_16_160_160 \
--visual_dir=/h3cstore_ns/hyshi/EM_mamba_new/result/superhuman_ac3_lr5_b2_16_160_160/visual \
--log_dir=/h3cstore_ns/hyshi/EM_mamba_new/result/superhuman_ac3_lr5_b2_16_160_160/tensorboard_log \
--warmup_epochs=0 --blr=1e-5 &
wait
echo "All tasks completed."