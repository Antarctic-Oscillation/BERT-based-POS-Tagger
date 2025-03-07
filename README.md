# BERT-based POS Tagger

A Part-of-Speech (POS) tagger built using Transformer models from Hugging Face, trained on the Universal Dependencies English Web Treebank (EWT) dataset.

## Overview

This project implements a neural POS tagger using pre-trained language models. It supports both DistilBERT (smaller, faster) and BERT-large models for tagging words with their correct part of speech according to the Universal Dependencies schema.

## Features

- Utilizes pre-trained transformer models (DistilBERT/BERT) for POS tagging
- Trained on the Universal Dependencies English Web Treebank dataset
- Handles tokenization alignment and subword tokenization challenges
- Includes comprehensive evaluation metrics and visualizations
- Supports model checkpointing and resumption of training
- Handles token-label alignment for transformer-based models

## Installation

```bash
# Clone the repository
git clone https://github.com/Antarctic-Oscillation/bert_parts-of-speech_tagger.git
cd bert_parts-of-speech_tagger

# Install requirements
pip install -r requirements.txt
```

## Usage

### Training the Model

The main script handles dataset loading, preprocessing, training, and evaluation:

You can modify the configuration variables at the top of the script:
- `SMALL_BERT`: Set to `True` to use DistilBERT (faster) or `False` to use BERT-large (more accurate)
- `MODEL_NAME`: Base pre-trained model to use
- `OUTPUT_DIR`: Directory for saving checkpoints
- `FINAL_SAVE_DIR`: Directory for the final model

### Using the Trained Model

from the predict.py script:
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json

model_path = "distilbert-base-cased-pos-tagger-final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

with open(f"{model_path}/label_mappings.json", "r") as f:
    mappings = json.load(f)
    id2label = mappings["id2label"]
    # Convert string keys back to integers
    id2label = {int(k): v for k, v in id2label.items()}

sentence = "I love coding with transformers"
tokens = sentence.split()
inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=2)

word_ids = inputs.word_ids()
previous_word_idx = None
pos_tags = []

for idx, word_idx in enumerate(word_ids):
    if word_idx is None or word_idx == previous_word_idx:
        continue
    pos_tags.append(id2label[predictions[0][idx].item()])
    previous_word_idx = word_idx

for token, tag in zip(tokens, pos_tags):
    print(f"{token}: {tag}")
```

## Training Details

- Uses the Universal Dependencies English Web Treebank dataset
- Training for 10 epochs with early stopping based on validation accuracy
- Learning rate: 2e-5
- Batch size: 16
- Weight decay: 0.01

## Performance

The model achieves good accuracy on the Universal Dependencies test set. The confusion matrix and performance metrics (accuracy, precision, recall, F1) are calculated in the jupyter notebook.

## Acknowledgments

- HuggingFace for the Transformers library
- Universal Dependencies project for the English Web Treebank dataset
