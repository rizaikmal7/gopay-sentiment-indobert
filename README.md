# GoPay Sentiment Analysis with IndoBERT

Fine-tuning IndoBERT for sentiment classification of Indonesian GoPay e-wallet tweets, with LLM-assisted labeling and data augmentation pipeline.

## Results

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Negatif | 78.6% | 73.3% | 75.9% |
| Netral | 87.7% | 89.9% | 88.8% |
| Positif | 87.1% | 90.0% | 88.5% |

**Overall Accuracy: 87.6% | Macro F1: ~85%**

## Project Structure

```
gopay-sentiment-indobert/
│
├── data/
│   ├── gopay_raw.csv              # 481 raw tweets (original)
│   ├── gopay_labeled.csv          # after cleansing + LLM labeling
│   └── gopay_augmented.csv        # final dataset after augmentation (~750 samples)
│
├── notebooks/
│   ├── 1_label_groq.ipynb         # text cleansing + LLM-assisted labeling (Groq API)
│   ├── 2_augmentasi_groq.ipynb    # paraphrase augmentation for minority classes
│   └── 3_indobert_finetuning.ipynb # IndoBERT fine-tuning with class weight
│
├── confusion_matrix.png
├── training_curves.png
├── requirements.txt
└── README.md
```

## Pipeline Overview

```
gopay_raw.csv (481 tweets)
    │
    ▼
Text Cleansing          → remove URL, mention, hashtag symbol, punctuation, numbers
    │
    ▼
LLM Labeling            → Groq API · LLaMA 3.3-70B · temperature=0.0
    │                     labels: positif / negatif / netral
    ▼
Data Augmentation       → Groq API · LLaMA 3.1-8B-Instant · temperature=0.8
    │                     paraphrase minority classes to ≥150 samples each
    ▼
IndoBERT Fine-tuning    → indobenchmark/indobert-base-p1
    │                     CrossEntropyLoss + balanced class weight
    │                     5 epochs · AdamW lr=2e-5 · batch size 16
    ▼
Evaluation              → Accuracy 87.6% · Macro F1 ~85%
```

## Dataset

- **Source:** Twitter/X — GoPay e-wallet Indonesian tweets
- **Raw size:** 481 tweets
- **Labeling:** LLM-assisted using Groq API (LLaMA 3.3-70B) at `temperature=0.0`
- **Label distribution (before augmentation):**

  | Label | Count |
  |-------|-------|
  | Netral | 390 |
  | Positif | 55 |
  | Negatif | 36 |

- **After augmentation:** ~750 samples, balanced to ≥150 per minority class

## Setup

```bash
pip install transformers==4.36.2 tokenizers==0.15.2 torch pandas \
    scikit-learn seaborn matplotlib groq
```

> After installing, **restart the runtime** before running notebooks 2 and 3.

## Usage

Run notebooks in order:

```
1_label_groq.ipynb          → produces gopay_labeled.csv
2_augmentasi_groq.ipynb     → produces gopay_augmented.csv
3_indobert_finetuning.ipynb → produces model + evaluation outputs
```

Set your Groq API key in Cell 2 of notebooks 1 and 2:
```python
GROQ_API_KEY = 'your_api_key_here'
```

Get a free API key at [console.groq.com](https://console.groq.com)

## Model

- **Base model:** [indobenchmark/indobert-base-p1](https://huggingface.co/indobenchmark/indobert-base-p1)
- **Task:** Sequence classification (3 labels)
- **Training:** 5 epochs, AdamW lr=2e-5, linear warmup 10%, batch size 16
- **Loss:** `CrossEntropyLoss` with balanced class weights

## Tech Stack

`Python` `PyTorch` `Hugging Face Transformers` `IndoBERT` `Groq API` `LLaMA 3.3-70B` `LLaMA 3.1-8B-Instant` `scikit-learn` `pandas` `Google Colab`

## Author

**Muhammad Ikmal Riza** — Informatics, Siliwangi University  
[github.com/rizaikmal7](https://github.com/rizaikmal7) · [LinkedIn](https://www.linkedin.com/in/muhammad-ikmal-riza)
