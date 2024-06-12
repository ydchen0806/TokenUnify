pip3 uninstall -y timm &&
pip3 install timm==0.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple  &&
sed -i 's/from torch._six import container_abcs/import collections.abc as container_abcs/g' /usr/local/lib/python3.8/dist-packages/timm/models/layers/helpers.py &&
export NCCL_P2P_DISABLE=1 &&
python3 -m torch.distributed.launch --nproc_per_node=8 /h3cstore_ns/hyshi/EM_mamba_new/EM_mamba_seg/main_finetune2.py --batch_size=20 \
--epochs=800 --output_dir=/h3cstore_ns/hyshi/EM_mamba_new/result/superhuman_monai_wafer_lr5_b20_18_160_160_gaussian_8gpu \
--visual_dir=/h3cstore_ns/hyshi/EM_mamba_new/result/superhuman_monai_wafer_lr5_b20_18_160_160_gaussian_8gpu/visual \
--log_dir=/h3cstore_ns/hyshi/EM_mamba_new/result/superhuman_monai_wafer_lr5_b20_18_160_160_gaussian_8gpu/tensorboard_log \
--warmup_epochs=0 --blr=1e-5