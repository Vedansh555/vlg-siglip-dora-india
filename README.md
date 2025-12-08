# Bridging the Cultural Divide: SigLIP + DoRA for Indian Contexts

**Status:** Research Proposal (VLG Track)  
**Author:** Vedansh Singh Gautam  
**Focus:** Cultural Domain Adaptation, Parameter-Efficient Fine-Tuning (PEFT)

## 📌 Overview
This project addresses the "Cultural Alignment Gap" in modern Vision-Language Models. While models like CLIP perform well globally, they struggle with region-specific concepts (e.g., distinguishing a *Sherwani* from a *Coat*).

We propose replacing standard approaches with a cutting-edge pipeline:
1.  **Architecture:** Google's **SigLIP** (Sigmoid Loss for Language Image Pre-Training) instead of the older CLIP.
2.  **Method:** **DoRA** (Weight-Decomposed LoRA) for robust fine-tuning without catastrophic forgetting.
3.  **Benchmark:** Introduction of **Indi-Bench**, a "Hard-Negative" evaluation set targeting high-confusion Indian cultural pairs.

## 🚀 Key Innovations
| Feature | Standard Approach | Our Approach |
| :--- | :--- | :--- |
| **Model** | CLIP (ViT-B/32) | **SigLIP (ViT-B/16)** |
| **Fine-Tuning** | Vanilla LoRA | **DoRA (Weight-Decomposed)** |
| **Evaluation** | General Accuracy | **Hard-Negative Cultural F1-Score** |

## 📂 Repository Structure
```bash
├── data/
│   ├── indi_bench/       # The custom hard-negative test set
│   └── raw/              # Scripts to download DollarStreet/Food101 subsets
├── src/
│   ├── config.py         # DoRA and SigLIP configuration
│   ├── train.py          # Main training loop (HuggingFace PEFT)
│   └── eval.py           # Evaluation script for Indi-Bench
├── notebooks/
│   └── 01_baseline.ipynb # Zero-shot inference demo
└── README.md
