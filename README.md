# Fine-Tuning Microsoft Phi-2 Using LoRA for Instruction-Following Tasks

This project demonstrates how to fine-tune Microsoft's Phi-2 large language model using LoRA (Low-Rank Adaptation) to generate structured outputs from natural language commands. The model is trained to follow instructions and output predefined JSON-based formats suitable for downstream robotic control or automation tasks.

---

## 📁 Repository Structure

.
├── output_phi2_lora/ # Directory containing final model and checkpoint files
├── split_output/ # Directory with tokenized train/validation datasets
├── split.py # Script to split raw dataset into train/validation sets
├── train.py # Main training script using Hugging Face + PEFT
└── README.md # Project documentation


---

## Getting Started

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

2. Prepare Dataset

Place your .jsonl file in the working directory, then use split.py to divide it:

python split.py

3. Run Training

python train.py

Trained model artifacts will be saved to the output_phi2_lora/ directory.
🔧 Model Overview

    Base Model: microsoft/phi-2

    Adapter Method: LoRA (r=4, alpha=16, dropout=0.05)

    Task: Instruction-following with structured JSON output

👥 Contributors

    Muhammad Azka Bintang Pramudya

    Mukesh Sadhasivam

    Sahil Abdullayev


AI Project 2
University of Toulon
EMJMD MIR (Marine and Maritime Intelligent Robotics)
May 21, 2025