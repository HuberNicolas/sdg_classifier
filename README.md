# sdg-classifier

> A Python library for SDG classification of scientific texts using BERT-based models and LoRA fine-tuning, developed in the context of the master's thesis.

## ✨ TL;DR
```bash
poetry install
poetry run python train.py
```

## 🚀 Features
- **BERT + LoRA:** Fine-tuning of transformer models via PEFT/LoRA for SDG multi-label classification.
- **BERTopic:** Topic modeling and keyword extraction for publication texts.
- **Modular:** Separate modules for preprocessing (`preparation.py`), training (`train.py`, `train_lora.py`), and evaluation.

## 🎓 Academic Context
This project was created as part of a master's thesis.
- **Course:** Master's Thesis in Informatics – Interactive Gamified System for AI-Assisted SDG Labeling in Scientific Research
- **Institution:** University of Zurich (UZH)
- **Semester:** Spring Semester 2025

---

## 🛠️ Installation

### Prerequisites
- Python 3.10.14
- [Poetry](https://python-poetry.org/)
- CUDA-capable GPU (recommended for training)

### Steps
1. Clone the repository:
```bash
git clone https://github.com/HuberNicolas/sdg_classifier
cd sdg_classifier
```
2. Install dependencies:
```bash
poetry install
```

---

## 💻 Usage

```bash
# Prepare data
poetry run python preparation.py

# Train the model (standard)
poetry run python train.py

# Train the model with LoRA
poetry run python train_lora.py
```

---

## 📄 License

This project is licensed under the **GNU General Public License v3 (GPLv3)** – see the [LICENSE](LICENSE) file for details.

**Conditions:**
- The original copyright notice (crediting my name) must be retained in all copies or substantial portions of the software.
- Modifications and derivative works *must* also be released under the GPLv3, and the source code must be made publicly available.
