from transformers import AutoTokenizer, AutoModelForTokenClassification
import json

model_path = "distilbert-base-cased-pos-tagger-final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

with open(f"{model_path}/label_mappings.json", "r") as f:
    mappings = json.load(f)
    id2label = mappings["id2label"]
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