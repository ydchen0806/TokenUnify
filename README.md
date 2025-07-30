# TokenUnify: Scaling Up Autoregressive Pretraining for Neuron Segmentation

**Yinda Chen¹,²\***, **Haoyuan Shi¹,²\***, **Xiaoyu Liu¹**, **Te Shi²**, **Ruobing Zhang³,²**, **Dong Liu¹**, **Zhiwei Xiong¹,²†**, **Feng Wu¹,²‡**

¹*MoE Key Laboratory of Brain-inspired Intelligent Perception and Cognition, University of Science and Technology of China*  
²*Institute of Artificial Intelligence, Hefei Comprehensive National Science Center*  
³*Institute for Brain and Intelligence, Fudan University*

\**Equal Contribution* †*Project Leader* ‡*Corresponding Author*

---

This repository contains the official implementation of the paper **[TokenUnify: Scaling Up Autoregressive Pretraining for Neuron Segmentation](https://arxiv.org/pdf/2405.16847)**. It includes experimental settings, source code, and theoretical proofs. For details, please refer to the [original paper](https://arxiv.org/pdf/2405.16847).

<div style="text-align: center;">
  <img src="framework1.png" alt="Pipeline of TokenUnify" width="80%" />
  <p><i>Pipeline of our proposed methods</i></p>
</div>

<div style="text-align: center;">
  <img src="framework2.png" alt="Network details of TokenUnify" width="80%" />
  <p><i>Network details of our proposed methods</i></p>
</div>

## 📰 News

- **[2025.06] 🎉 TokenUnify** was accepted by ICCV 2025, looking forward to meeting you in Hawaii.
- **[2025.06] 📊 MEC dataset released!** Wafer (MEC) dataset available on [HuggingFace](https://huggingface.co/datasets/cyd0806/wafer_EM).
- **[2025.06] 🔧 Pre-trained weights updated!** Robust initialization weights (pre-trained, not fine-tuned) available in the [Pretrained_weights folder](https://huggingface.co/cyd0806/TokenUnify/tree/main/Pretrained_weights) on HuggingFace.
- **[2024.12] 🎉 Code and pre-training dataset released!** Core implementation and pre-training weights released.
- **[2024.12] 📊 Datasets released!** Pre-training dataset available on [HuggingFace](https://huggingface.co/datasets/cyd0806/EM_pretrain_data).
- **[2024.05] 📝 Paper released!** TokenUnify paper published on [arXiv](https://arxiv.org/pdf/2405.16847).

## 🚀 Overview

TokenUnify introduces a novel autoregressive visual pre-training method for neuron segmentation from electron microscopy (EM) volumes. The method tackles the unique challenges of EM data including high noise levels, anisotropic voxel dimensions, and ultra-long spatial dependencies through hierarchical predictive coding that combines three complementary prediction tasks:

- **Random Token Prediction**: Captures noise-robust spatial patterns and learns position-invariant local feature detectors.
- **Next Token Prediction**: Maintains sequential dependencies and captures critical transitional patterns in neuronal morphology.
- **Next-All Token Prediction**: Models global context and long-range correlations while mitigating cumulative errors in autoregression.

Leveraging the Mamba architecture's linear-time sequence modeling capabilities, TokenUnify achieves **44% improvement** in neuron segmentation performance compared to training from scratch and **25% improvement** over MAE, while demonstrating superior scaling properties and reducing autoregressive error accumulation from O(K) to O(√K) for sequences of length K.

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

## 🔬 Key Technical Contributions

1. **Hierarchical Predictive Coding Framework**: We introduce a unified framework that integrates three distinct visual structure perspectives within a coherent information-theoretic formulation, providing optimal coverage of visual data structure while reducing autoregressive error accumulation from O(K) to O(√K).

2. **Large-Scale EM Dataset**: We construct one of the largest EM neuron segmentation datasets with 1.2 billion finely annotated voxels across six functional brain regions, providing an ideal testbed for long-sequence visual modeling.

3. **Billion-Parameter Mamba Network**: We achieve the first billion-parameter Mamba network for visual autoregression, demonstrating both effectiveness and computational efficiency in processing long-sequence visual data with favorable scaling properties.

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
- **Authors:** Yinda Chen, Haoyuan Shi, Xiaoyu Liu, Te Shi, Ruobing Zhang, Dong Liu, Zhiwei Xiong, Feng Wu
- **Copyright Holder:** Institute of Artificial Intelligence, Hefei Comprehensive National Science Center

### **Acknowledgment Norms**

- **Chinese Name:** 合肥人工智能研究院
- **English Name:** Institute of Artificial Intelligence, Hefei Comprehensive National Science Center

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
@inproceedings{chen2025tokenunify,
  title={TokenUnify: Scaling Up Autoregressive Pretraining for Neuron Segmentation},
  author={Chen, Yinda and Shi, Haoyuan and Liu, Xiaoyu and Shi, Te and Zhang, RuoBing and Liu, Dong and Xiong, Zhiwei and Wu, Feng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions to improve TokenUnify! Please submit issues and pull requests.

## 📧 Contact

For questions, contact: `cyd0806@mail.ustc.edu.cn`

---

⭐ **If you find this work helpful, please give us a star!** ⭐
