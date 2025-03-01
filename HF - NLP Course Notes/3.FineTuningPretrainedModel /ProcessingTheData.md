### **📌 Processing Data for Fine-Tuning a Pretrained Transformer Model**

---

## **1️⃣ Overview: Fine-Tuning a Sequence Classifier**

Fine-tuning a Transformer model involves **preprocessing data**, **tokenizing text**, **handling batch processing**, and **applying dynamic padding** for efficiency.

✔ **Dataset Used:** **MRPC (Microsoft Research Paraphrase Corpus)**

✔ **Task:** Classify whether two sentences are **paraphrases** or not.

✔ **Model Used:** **BERT (`bert-base-uncased`)**

🚀 **Goal:** Train a Transformer model on **paraphrase detection** using optimized data handling techniques.

---

## **2️⃣ Loading a Dataset from the 🤗 Hub**

💡 **🤗 Datasets** provides a simple API to **load & process datasets** from the Hugging Face Hub.

✅ **Install & Load the MRPC Dataset**

```python
from datasets import load_dataset

# Load the dataset from the Hugging Face Hub
raw_datasets = load_dataset("glue", "mrpc")

# Check dataset structure
print(raw_datasets)

```

✔️ **Dataset contains:**

- **Train:** 3,668 pairs
- **Validation:** 408 pairs
- **Test:** 1,725 pairs

📌 **Accessing a Specific Example**

```python
raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset[0])

```

✔️ **Sample Output:**

```python
{
 'idx': 0,
 'label': 1,
 'sentence1': 'Amrozi accused his brother, whom he called "the witness", of deliberately distorting his evidence.',
 'sentence2': 'Referring to him as only "the witness", Amrozi accused his brother of deliberately distorting his evidence.'
}

```

✔ **Labels:**

- `0` → **Not a paraphrase**
- `1` → **Paraphrase**

📌 **Check Dataset Features**

```python
print(raw_train_dataset.features)

```

✔ **Identifies each feature type (string, label, int, etc.)**

---

## **3️⃣ Tokenizing the Dataset**

💡 **Why Tokenize?**

✔️ Converts text into **numerical representations**.

✔️ Splits sentences into **tokens** and assigns **unique IDs**.

✔️ Handles **sentence pairs** for BERT.

✅ **Initialize Tokenizer**

```python
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

```

### **🔹 Handling Sentence Pairs for BERT**

💡 **BERT needs `[CLS] sentence1 [SEP] sentence2 [SEP]` format.**

✅ **Example Tokenization**

```python
inputs = tokenizer("This is the first sentence.", "This is the second one.")
print(inputs)

```

✔ **Tokenized Output**

```python
{
 'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

```

✔ **`token_type_ids` (segment IDs):**

- `0`: **First sentence**
- `1`: **Second sentence**

📌 **Decoding Input IDs**

```python
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))

```

✔ Converts tokens **back to readable text.**

---

## **4️⃣ Preprocessing the Entire Dataset**

✅ **Tokenizing the Full Dataset**

```python
tokenized_datasets = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)

```

✔ **Challenges:**

- Requires **high memory**.
- Returns **Python lists, not efficient datasets**.

✅ **Better Approach: Using `map()` for Efficient Tokenization**

```python
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)

```

✔ **Faster & More Memory-Efficient**

✔ **Stores tokenized dataset in Apache Arrow format (optimized for disk access)**

---

## **5️⃣ Handling Batching with Dynamic Padding**

💡 **Why Dynamic Padding?**

✔ **Optimized training** → Only pad to the longest sentence in a batch, not entire dataset.

✔ **Avoids excessive computation on padding tokens.**

✅ **Use `DataCollatorWithPadding` to Enable Dynamic Padding**

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

```

✅ **Test Dynamic Padding**

```python
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}

# Check input lengths before padding
print([len(x) for x in samples["input_ids"]])

# Apply dynamic padding
batch = data_collator(samples)

# Check final batch shape
print({k: v.shape for k, v in batch.items()})

```

✔ **Dynamically Pads to Longest Sequence in Each Batch**

---

## **6️⃣ Full Workflow: Fine-Tuning BERT on MRPC**

✅ **Complete Code: Data Preprocessing, Tokenization, and Batching**

```python
import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset

# Load dataset
raw_datasets = load_dataset("glue", "mrpc")

# Load tokenizer & model
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# Tokenization function
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

# Tokenize dataset
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# Define data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Select samples and apply dynamic padding
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
batch = data_collator(samples)

# Check batch structure
print({k: v.shape for k, v in batch.items()})

# Fine-tuning step
batch["labels"] = torch.tensor([1, 1])  # Example labels
optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()

```

✔ **Complete workflow:**

✅ **Loads dataset**

✅ **Tokenizes & batches data**

✅ **Applies dynamic padding**

✅ **Trains the model**

---

## **🎯 Summary – Key Takeaways**

✔ **Use `load_dataset()` to load datasets efficiently from the 🤗 Hub.**

✔ **Tokenize efficiently with `map()` instead of processing everything in memory.**

✔ **For sentence pairs, use `[CLS] sentence1 [SEP] sentence2 [SEP]` format.**

✔ **Use `DataCollatorWithPadding` for dynamic padding → Saves memory & speeds up training.**

✔ **Use `AdamW` optimizer & compute loss during training.**

🚀 **Next Steps: Creating DataLoaders & Training Loops for Full Fine-Tuning!**