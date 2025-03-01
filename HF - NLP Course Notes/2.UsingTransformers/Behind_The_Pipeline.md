### **🔍 Behind the Pipeline – How Transformers Work Step by Step**

---

## **1️⃣ The Pipeline Process**

💡 The **`pipeline()` function** in 🤗 Transformers simplifies NLP tasks into **three key steps**:

✔️ **Preprocessing** → Convert raw text into numerical input using a **tokenizer**.

✔️ **Model Inference** → Pass the processed inputs through a **Transformer model**.

✔️ **Postprocessing** → Convert model outputs into human-readable predictions.

✅ **Example: Sentiment Analysis Pipeline**

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier(["I've been waiting for a Hugging Face course my whole life.", "I hate this so much!"])

```

📌 **Output:**

```json
[{'label': 'POSITIVE', 'score': 0.9598},
 {'label': 'NEGATIVE', 'score': 0.9995}]

```

---

## **2️⃣ Step 1: Preprocessing with a Tokenizer**

📌 **Why?**

- Transformer models **can't process raw text** → Need conversion into **tokens & numerical inputs**.
- Uses **`AutoTokenizer.from_pretrained()`** to load the correct tokenizer.

✅ **Example: Tokenizing Sentences**

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a Hugging Face course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")

```

📌 **Output:**

```json
{
  'input_ids': [[101, 1045, 1005, 2310, ..., 1012, 102], [101, 1045, 5223, ..., 999, 102, 0, 0, 0]],
  'attention_mask': [[1, 1, 1, ..., 1, 1], [1, 1, 1, ..., 1, 1, 0, 0, 0]]
}

```

✔️ **`input_ids`** → Tokenized text converted into numerical values.

✔️ **`attention_mask`** → Tells the model which tokens to **attend to (1)** and **ignore (0)**.

---

## **3️⃣ Step 2: Model Inference (Passing Inputs to the Model)**

📌 **Load a Pretrained Model**

- Uses **`AutoModel.from_pretrained()`** to load the base Transformer model.
- Outputs **hidden states** (high-dimensional contextual representations).

✅ **Example: Load Model & Run Inference**

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)  # Output shape

```

📌 **Output:**

```json
torch.Size([2, 16, 768])  # (Batch size, Sequence length, Hidden size)

```

✔️ **Batch size = 2** → Two input sentences.

✔️ **Sequence length = 16** → Tokens per sentence (with padding).

✔️ **Hidden size = 768** → Each token is represented as a **768-dimensional vector**.

---

## **4️⃣ Step 3: Adding a Model Head for Classification**

📌 **Why?**

- The base model **only generates embeddings**.
- To classify text, we need a **task-specific model head** (e.g., SequenceClassification).

✅ **Load Model with a Classification Head**

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.logits.shape)  # Shape of the output logits

```

📌 **Output:**

```json
torch.Size([2, 2])  # (Batch size, Number of labels)

```

✔️ **Two output values per sentence (POSITIVE & NEGATIVE scores).**

---

## **5️⃣ Step 4: Postprocessing (Converting Logits to Probabilities)**

📌 **Raw model outputs (logits) are unnormalized → Apply SoftMax to get probabilities.**

✅ **Example: Convert Logits to Probabilities**

```python
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

```

📌 **Output:**

```json
tensor([[0.0402, 0.9598],  # Sentence 1: 4% Negative, 96% Positive
        [0.9995, 0.0005]]) # Sentence 2: 99.95% Negative, 0.05% Positive

```

📌 **Map Output to Labels:**

```python
model.config.id2label

```

📌 **Output:**

```json
{0: 'NEGATIVE', 1: 'POSITIVE'}

```

✅ **Final Predictions:**

- **Sentence 1:** **POSITIVE (96%)**
- **Sentence 2:** **NEGATIVE (99.95%)**

---

### **🎯 Summary – What Happens Behind the Pipeline?**

1️⃣ **Preprocessing** → **Tokenization** (Convert text → token IDs).

2️⃣ **Model Inference** → **Pass input through Transformer model** to get hidden states.

3️⃣ **Adding a Model Head** → **Convert hidden states to classification scores**.

4️⃣ **Postprocessing** → **Apply SoftMax** to get probabilities & map to labels.

✅ **We have successfully replicated the `pipeline()` function manually!** 🚀

---

### **✏️ Try It Out!**

1️⃣ Choose **two sentences** of your own.

2️⃣ Run them through the **sentiment-analysis pipeline**.

3️⃣ Manually replicate each step to verify the results! 🔥
