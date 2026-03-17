# 🌍 Mitigating Western Bias in Vision-Language Models

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F9D423)

## 📌 Project Overview
Foundational Vision-Language Models (VLMs) exhibit a pronounced "Western Bias" due to their training data. When presented with specific regional artifacts—particularly traditional Indian garments—these models often default to generic Western classifications (e.g., misclassifying a *Lehenga* as a *Ball Gown*).

This repository contains the code and methodology to:
1. **Automate the generation** of a high-fidelity "Cultural Misfit Benchmark" using Image-to-Image models.
2. **Evaluate and quantify** the zero-shot failure rates of SOTA open-weights VLMs using a 2-Alternative Forced Choice (2AFC) methodology.
3. **Mitigate the bias** by adapting contrastive models (like Google SigLIP) to the Indian context using Weight-Decomposed Low-Rank Adaptation (DoRA).

---

## 🛠️ The Hard-Negative Generation Pipeline
To mathematically prove this bias, models must be tested on "Hard Negatives"—images that share the same visual silhouette but belong to a different cultural context. 

We built an automated, VRAM-efficient pipeline to generate these pairs from the Indo-Fashion dataset:
1. **Context Extraction:** `Qwen2.5-VL-7B` analyzes the original Indian garment and generates a highly specific editing prompt for a Western equivalent.
2. **Localized Editing:** `FLUX.1-schnell` (quantized) performs localized inpainting to seamlessly transform the Indian garment into the Western trap label while preserving the original background and subject pose.

*Result:* A pristine, 150-image paired benchmark dataset.

---

## 📊 Benchmark Results (The 2AFC Showdown)
Models were presented with the Indian garment and the generated Western garment side-by-side and prompted: *"Which side is the [Indian Garment] on?"* A score near 50% indicates a complete lack of latent cultural representation (blind guessing).

| Model Architecture | Accuracy | Western Bias (Error Rate) | Verdict |
| :--- | :--- | :--- | :--- |
| **LLaVA-1.5-7B** | 50.0% | 50.0% | ❌ Complete random guessing. |
| **Qwen2-VL-2B** | 47.5% | 52.5% | ❌ High bias. Favors Western concepts. |
| **BLIP-2 (flan-t5-xl)** | 42.5% | 57.5% | ❌ Severe bias. Actively misclassifies. |

---

## 📂 Repository Structure

```text
├── notebooks/
│   ├── 01_flux_hard_negative_generation.ipynb   # Pipeline for creating the dataset
│   ├── 02_vlm_2afc_evaluation.ipynb             # Testing script for LLaVA, Qwen, etc.
│   └── 03_siglip_dora_finetuning.ipynb          # Mitigation via DoRA
├── README.md
└── requirements.txt
