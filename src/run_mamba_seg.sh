pip3 uninstall -y timm &&
pip3 install timm==0.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple &&
sed -i 's/from torch._six import container_abcs/import collections.abc as container_abcs/g' /usr/local/lib/python3.8/dist-packages/timm/models/layers/helpers.py &&
export NCCL_P2P_DISABLE=1 &&
export NCCL_SOCKET_TIMEOUT=3600 &&
python3 -m torch.distributed.launch --nproc_per_node=8 /h3cstore_ns/hyshi/EM_mamba_new/EM_mamba_seg/main_finetune.py --batch_size=6 \
--epochs=400 \
--warmup_epochs=0 --blr=1e-4



# sed -i 's/from torch._six import container_abcs/import collections.abc as container_abcs/g' /public/home/heart_llm/.local/lib/python3.10/site-packages/timm/models/layers/helpers.py &&
# sed -i 's/from torch._six import container_abcs/import collections.abc as container_abcs/g' /public/home/heart_llm/.local/lib/python3.8/site-packages/timm/models/layers/helpers.py &&
