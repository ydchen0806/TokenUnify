# TokenUnify: Scalable Autoregressive Visual Pre-training with Mixture Token Prediction

This repository contains the official implementation of the paper **[TokenUnify: Scalable Autoregressive Visual Pre-training with Mixture Token Prediction](https://arxiv.org/pdf/2405.16847)**. It provides all the experimental settings and source code used in our research. The paper also includes theoretical proofs. For more details, please refer to our original paper.

<div style="text-align: center;">
  <img src="framework1.png" alt="The pipeline of our proposed methods" width="80%" />
</div>

<div style="text-align: center;">
  <img src="framework2.png" alt="The network details of our proposed methods" width="80%" />
</div>

## 📰 News

- **[2024.12] 🎉 Code and pre-training dataset released!** Core implementation and pre-training weights are now available.
- **[2024.12] 📊 Datasets released!** Pre-training dataset available on [HuggingFace](https://huggingface.co/datasets/cyd0806/EM_pretrain_data).
- **[2024.12] 🔧 Pre-trained weights released!** Model weights available at [HuggingFace](https://huggingface.co/cyd0806/TokenUnify).
- **[2024.05] 📝 Paper released!** TokenUnify paper published on [arXiv](https://arxiv.org/pdf/2405.16847).

## 🚀 Overview

TokenUnify introduces a novel autoregressive visual pre-training method that integrates three distinct prediction tasks:
- **Random Token Prediction**: Captures global contextual information
- **Next Token Prediction**: Maintains sequential dependencies  
- **Next-All Token Prediction**: Mitigates cumulative errors in autoregression

Our approach demonstrates superior scaling laws and achieves **45% improvement** in neuron segmentation performance while reducing computational complexity through the Mamba architecture.

## 🛠️ Environment Setup

To streamline the setup process, we provide a Docker image that can be used to set up the environment with a single command:

```bash
sudo docker pull registry.cn-hangzhou.aliyuncs.com/mybitahub/large_model:mamba0224_ydchen
```

## 📦 Dataset Download

The datasets required for pre-training and segmentation are as follows:

| Dataset Type          | Dataset Name           | Description                              | URL                                           |
|-----------------------|------------------------|------------------------------------------|-----------------------------------------------|
| Pre-training Dataset  | Large EM Datasets | Various brain regions for pre-training | [🤗 EM Pretrain Dataset](https://huggingface.co/datasets/cyd0806/EM_pretrain_data) |
| Segmentation Dataset  | CREMI Dataset          | Challenge on circuit reconstruction | [CREMI Dataset](https://cremi.org/) |
| Segmentation Dataset  | AC3/AC4 | Mouse brain cortex dataset | [Google Drive](https://drive.google.com/drive/folders/1JAdoKchlWrHnbTXvnFn6pWWwx6VIiMH3?usp=sharing) |
| Segmentation Dataset  | MEC | High-resolution neuron segmentation | *Coming soon* |

## 🏋️ Model Weights

Pre-trained TokenUnify weights are now available:

| Model | Parameters | Dataset | URL |
|-------|------------|---------|-----|
| TokenUnify-Base | 28M | EM Multi-dataset | [🤗 HuggingFace](https://huggingface.co/cyd0806/TokenUnify) |
| TokenUnify-Large | 500M | EM Multi-dataset | [🤗 HuggingFace](https://huggingface.co/cyd0806/TokenUnify) |
| TokenUnify-Huge | 1B | EM Multi-dataset | [🤗 HuggingFace](https://huggingface.co/cyd0806/TokenUnify) |

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

### 1. Scaling Law of TokenUnify
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

- [x] 📝 Open-sourced the core code
- [x] 📖 Wrote the README for code usage  
- [x] 🗂️ Open-sourced the pre-training dataset
- [x] ⚖️ Upload the pre-trained weights
- [ ] 🧠 Release the private dataset MEC
- [ ] 🏆 Release evaluation scripts and benchmarks
- [ ] 🔧 Add support for natural image datasets

## 📝 Citation

If you find this code or dataset useful in your research, please consider citing our paper:

```bibtex
@article{chen2024tokenunify,
  title={TokenUnify: Scalable Autoregressive Visual Pre-training with Mixture Token Prediction},
  author={Chen, Yinda and Shi, Haoyuan and Liu, Xiaoyu and Shi, Te and Zhang, RuoBing and Liu, Dong and Xiong, Zhiwei and Wu, Feng},
  journal={arXiv preprint arXiv:2405.16847},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions to improve TokenUnify! Please feel free to submit issues and pull requests.

## 📧 Contact

For questions or issues, please contact: `cyd0806@mail.ustc.edu.cn`

---

⭐ **If you find this work helpful, please consider giving us a star!** ⭐
