## Environment Setup

To streamline the setup process, we provide a Docker image that can be used to set up the environment with a single command. The Docker image is available at:

```sh
docker pull registry.cn-hangzhou.aliyuncs.com/cyd_dl/monai-vit:v26
```
## Dataset Download

The datasets required for pre-training and segmentation are as follows:

| Dataset Type          | Dataset Name           | Description                              | URL                                           |
|-----------------------|------------------------|------------------------------------------|-----------------------------------------------|
| Pre-training Dataset  | Region of FAFB Dataset | Fly brain dataset for pre-training       | [EM Pretrain Dataset](https://huggingface.co/datasets/cyd0806/EM_pretrain_data/tree/main)  |
| Segmentation Dataset  | CREMI Dataset          | Challenge on circuit reconstruction datasets| [CREMI Dataset](https://cremi.org/)           |

### Pre-training Dataset: Region of FAFB

The FAFB region dataset is used for pre-training. Please follow the instructions provided in the paper to acquire and preprocess this dataset. You can download it from the Hugging Face [EM Pretrain Dataset](https://huggingface.co/datasets/cyd0806/EM_pretrain_data/tree/main). Use the subfolder `FAFB_hdf` to match the paper's settings, or use additional relevant data to achieve better results.

To use this dataset, please refer to the license provided [here](#license-important-).

### Segmentation Dataset: CREMI

The CREMI dataset is used for the segmentation tasks. Detailed instructions for downloading and preprocessing can be found on the [CREMI Challenge website](https://cremi.org/).

## Usage Guide

### 1. Pretraining
```
python pretrain.py -c pretraining_all -m train
```
### 2. Finetuning
```
python finetune.py -c seg_3d -m train -w [your pretrained path]
```
