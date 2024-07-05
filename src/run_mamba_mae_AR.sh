pip3 uninstall -y timm &&
pip3 install timm==0.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple &&
sed -i 's/from torch._six import container_abcs/import collections.abc as container_abcs/g' /usr/local/lib/python3.8/dist-packages/timm/models/layers/helpers.py &&
# pip3 install --upgrade torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple &&
export NCCL_P2P_DISABLE=1 &&
export ROOT_DIR=/h3cstore_ns/EM_pretrain/mamba_pretrain_autoregress_0430_amp/0428segmamba_auto_mode${1}_fill_nose${2} &&
python3 -m torch.distributed.launch --nproc_per_node=8 /data/ydchen/VLP/EM_Mamba/mambamae_EM/main_pretrain_autoregress.py --batch_size=20 \
--epochs=800 --model=segmamba --use_amp=True --output_dir=$ROOT_DIR \
--visual_dir=$ROOT_DIR/visual --log_dir=$ROOT_DIR/tensorboard_log \
--warmup_epochs=0 --auto_mode=$1 --fill_nose=$2 --pretrain_path=$3 