# TokenUnify: Scalable Autoregressive Visual Pre-training with Mixture Token Prediction

This repository contains the official implementation of the paper **[TokenUnify: Scalable Autoregressive Visual Pre-training with Mixture Token Prediction](https://arxiv.org/pdf/2405.16847)**. It includes experimental settings, source code, and theoretical proofs. For details, please refer to the [original paper](https://arxiv.org/pdf/2405.16847).

<div style="text-align: center;">
  <img src="framework1.png" alt="Pipeline of TokenUnify" width="80%" />
  <p><i>Pipeline of our proposed methods</i></p>
</div>

<div style="text-align: center;">
  <img src="framework2.png" alt="Network details of TokenUnify" width="80%" />
  <p><i>Network details of our proposed methods</i></p>
</div>

## 📰 News

- **[2025.06] 📊 MEC dataset released!** Wafer (MEC) dataset available on [HuggingFace](https://huggingface.co/datasets/cyd0806/wafer_EM).
- **[2025.06] 🔧 Pre-trained weights updated!** Robust initialization weights (pre-trained, not fine-tuned) available in the [Pretrained_weights folder](https://huggingface.co/cyd0806/TokenUnify/tree/main/Pretrained_weights) on HuggingFace.
- **[2024.12] 🎉 Code and pre-training dataset released!** Core implementation and pre-training weights released.
- **[2024.12] 📊 Datasets released!** Pre-training dataset available on [HuggingFace](https://huggingface.co/datasets/cyd0806/EM_pretrain_data).
- **[2024.05] 📝 Paper released!** TokenUnify paper published on [arXiv](https://arxiv.org/pdf/2405.16847).

## 🚀 Overview

TokenUnify introduces a novel autoregressive visual pre-training method combining three prediction tasks:
- **Random Token Prediction**: Captures global contextual information.
- **Next Token Prediction**: Maintains sequential dependencies.
- **Next-All Token Prediction**: Mitigates cumulative errors in autoregression.

Leveraging the Mamba architecture, TokenUnify achieves **45% improvement** in neuron segmentation performance with reduced computational complexity and demonstrates superior scaling laws.

## 🛠️ Environment Setup

Set up the environment using our Docker image:

```bash
sudo docker pull registry.cn-hangzhou.aliyuncs.com/mybitahub/large_model:mamba0224_ydchen
```

## 📦 Dataset Download

Datasets for pre-training and segmentation:

| Dataset Type          | Dataset Name           | Description                              | URL                                           |
|-----------------------|------------------------|------------------------------------------|-----------------------------------------------|
| Pre-training Dataset  | Large EM Datasets      | Various brain regions for pre-training   | [🤗 EM Pretrain Dataset](https://huggingface.co/datasets/cyd0806/EM_pretrain_data) |
| Segmentation Dataset  | Wafer (MEC)            | High-resolution neuron segmentation      | [🤗 Wafer_EM Dataset](https://huggingface.co/datasets/cyd0806/wafer_EM) |
| Segmentation Dataset  | CREMI Dataset          | Circuit reconstruction challenge         | [CREMI Dataset](https://cremi.org/) |
| Segmentation Dataset  | AC3/AC4                | Mouse brain cortex dataset               | [Google Drive](https://drive.google.com/drive/folders/1JAdoKchlWrHnbTXvnF6pWWwx6VIiMH3?usp=sharing) |

## 🏋️ Model Weights

Pre-trained (robust initialization, not fine-tuned) TokenUnify weights are available in the [Pretrained_weights folder](https://huggingface.co/cyd0806/TokenUnify/tree/main/Pretrained_weights):

| Model                  | Parameters | Dataset           | URL                                                                 |
|------------------------|------------|-------------------|----------------------------------------------------------------------|
| TokenUnify_pretrained-100M | 100M   | EM Multi-dataset  | [🤗 Pretrained_weights](https://huggingface.co/cyd0806/TokenUnify/tree/main/Pretrained_weights) |
| TokenUnify_pretrained-200M | 200M   | EM Multi-dataset  | [🤗 Pretrained_weights](https://huggingface.co/cyd0806/TokenUnify/tree/main/Pretrained_weights) |
| TokenUnify_pretrained-500M | 500M   | EM Multi-dataset  | [🤗 Pretrained_weights](https://huggingface.co/cyd0806/TokenUnify/tree/main/Pretrained_weights) |
| TokenUnify_pretrained-1B   | 1B     | EM Multi-dataset  | [🤗 Pretrained_weights](https://huggingface.co/cyd0806/TokenUnify/tree/main/Pretrained_weights) |

Fine-tuned weights are also available:

| Model                  | Parameters | Dataset           | URL                                                                 |
|------------------------|------------|-------------------|----------------------------------------------------------------------|
| TokenUnify-100M        | 100M       | EM Multi-dataset  | [🤗 HuggingFace](https://huggingface.co/cyd0806/TokenUnify)          |
| TokenUnify-200M        | 200M       | EM Multi-dataset  | [🤗 HuggingFace](https://huggingface.co/cyd0806/TokenUnify)          |
| TokenUnify-500M        | 500M       | EM Multi-dataset  | [🤗 HuggingFace](https://huggingface.co/cyd0806/TokenUnify)          |
| TokenUnify-1B          | 1B         | EM Multi-dataset  | [🤗 HuggingFace](https://huggingface.co/cyd0806/TokenUnify)          |
| superhuman             | -          | EM Multi-dataset  | [🤗 HuggingFace](https://huggingface.co/cyd0806/TokenUnify)          |

## 🔥 Usage Guide

### 1. Pre-training (8 nodes)
```bash
bash src/run_mamba_mae_AR.sh
```

### 2. Pre-training (32 nodes - Large scale)
```bash
bash src/launch_huge.sh
```

### 3. Fine-tuning
```bash
bash src/run_mamba_seg.sh
```

## 📊 Results

### 1. Scaling Law
<div style="text-align: center;">
  <img src="results1.png" alt="Scaling Law of TokenUnify" width="80%" />
</div>

### 2. Main Results
<div style="text-align: center;">
  <img src="results2.png" alt="Main Results of TokenUnify" width="80%" />
</div>

### 3. Visual Results
<div style="text-align: center;">
  <img src="visual_results.png" alt="Visual Results of TokenUnify" width="80%" />
</div>

## 📄 License

<details>
<summary>Usage Notes</summary>

### **Before the public release of the data, the following usage restrictions must be met:**

1. **Non-commercial Use:** Users do not have the rights to copy, distribute, publish, or use the data for commercial purposes or develop and produce products. Any format or copy of the data is considered the same as the original data. Users may modify the content and convert the data format as needed but are not allowed to publish or provide services using the modified or converted data without permission.
   
2. **Research Purposes Only:** Users guarantee that the authorized data will only be used for their own research and will not share the data with third parties in any form.

3. **Citation Requirements:** Research results based on the authorized data, including books, articles, conference papers, theses, policy reports, and other publications, must cite the data source according to citation norms, including the authors and the publisher of the data.

4. **Prohibition of Profit-making Activities:** Users are not allowed to use the authorized data for any profit-making activities.

5. **Termination of Data Use:** Users must terminate all use of the data and destroy the data (e.g., completely delete from computer hard drives and storage devices/spaces) upon leaving their team or organization or when the authorization is revoked by the copyright holder.

### **Data Information**

- **Sample Source:** Mouse MEC MultiBeam-SEM, Intelligent Institute Brain Imaging Platform (Wafer 4 at layer VI, wafer 25, wafer 26, and wafer 36 at layer II/III)
- **Resolution:** 8nm x 8nm x 35nm
- **Volume Size:** 1250 x 1250 x 125
- **Annotation Completion Dates:** 2023.12.11 (w4), 2024.04.12 (w36)
- **Authors:** Anonymous authors
- **Copyright Holder:** Anonymous Agency

### **Acknowledgment Norms**

- **Chinese Name:** 匿名机构
- **English Name:** Anonymous Agency

</details>

## ✅ To-Do List

- [x] 📝 Open-source core code
- [x] 📖 Write README for code usage
- [x] 🗂️ Open-source pre-training dataset
- [x] ⚖️ Upload pre-trained and fine-tuned weights
- [x] 🧠 Release Wafer (MEC) dataset
- [ ] 🏆 Release evaluation scripts and benchmarks
- [ ] 🔧 Add support for natural image datasets

## 📝 Citation

If you find this code or dataset useful, please cite:

```bibtex
@article{chen2024tokenunify,
  title={TokenUnify: Scalable Autoregressive Visual Pre-training with Mixture Token Prediction},
  author={Chen, Yinda and Shi, Haoyuan and Liu, Xiaoyu and Shi, Te and Zhang, RuoBing and Liu, Dong and Xiong, Zhiwei and Wu, Feng},
  journal={arXiv preprint arXiv:2405.16847},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions to improve TokenUnify! Please submit issues and pull requests.

## 📧 Contact

For questions, contact: `cyd0806@mail.ustc.edu.cn`

---

⭐ **If you find this work helpful, please give us a star!** ⭐
