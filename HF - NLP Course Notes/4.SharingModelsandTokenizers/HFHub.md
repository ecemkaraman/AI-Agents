### **📌 Sharing Models & Tokenizers on the Hugging Face Hub**

---

## **1️⃣ Overview: What is the Hugging Face Hub?**

The **Hugging Face Hub** ([**huggingface.co**](https://huggingface.co/)) is a **centralized platform** where:

✔ **State-of-the-art ML models** are **shared, used, and versioned**.

✔ **Over 10,000+ models** are available **for NLP, vision, and audio** tasks.

✔ **Public models are free** (Private models require a paid plan).

✔ **Every model gets an automatic hosted **Inference API**.

🚀 **Goal:** Learn how to:

🔹 Upload a model to the Hugging Face Hub.

🔹 Share & version models with Git-like repositories.

🔹 Use & download models from the Hub.

---

## **2️⃣ Setting Up a Hugging Face Account**

💡 **To upload models, you need a Hugging Face account.**

✅ **Sign up on the Hugging Face Hub**

1. Go to [**huggingface.co/join**](https://huggingface.co/join).
2. Create an account (or log in if you already have one).
3. Generate an authentication token:
    - Navigate to **Settings → Access Tokens**.
    - Click **New Token**, set permissions to **Write**, and copy it.

✅ **Log in via CLI**

```python
from huggingface_hub import notebook_login

notebook_login()

```

✔ **Prompts you to paste your token**.

✔ **Grants access to upload models & datasets**.

✅ **Check Authentication**

```python
!huggingface-cli whoami

```

✔ **Verifies your connection to the Hugging Face Hub**.

---

## **3️⃣ Uploading a Model to the Hub**

💡 **Use `push_to_hub()` to upload models, tokenizers, and configs.**

✅ **Load & Fine-Tune a Model**

```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import torch

# Load dataset
dataset = load_dataset("glue", "mrpc")

# Load model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="bert-mrpc", push_to_hub=True, evaluation_strategy="epoch"
)

# Trainer
trainer = Trainer(model=model, args=training_args, train_dataset=dataset["train"])

```

✔ **Prepares model for training & Hub upload**

✅ **Push Model to the Hub**

```python
trainer.push_to_hub()

```

✔ **Automatically uploads model, tokenizer, and logs**

✔ **Creates a public repository at `huggingface.co/{username}/bert-mrpc`**

✅ **Manually Upload a Model**

```python
model.push_to_hub("bert-mrpc")

```

✔ **Uploads model separately from training artifacts**

✅ **Upload Tokenizer**

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.push_to_hub("bert-mrpc")

```

✔ **Uploads tokenizer configuration**

---

## **4️⃣ Using a Model from the Hub**

💡 **Once uploaded, models can be loaded directly from the Hub!**

✅ **Load the Model from the Hub**

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("username/bert-mrpc")

```

✔ **Downloads the model & weights automatically**

✅ **Load the Tokenizer**

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("username/bert-mrpc")

```

✔ **Ensures compatibility with model architecture**

✅ **Use Model for Inference**

```python
import torch

inputs = tokenizer("Hugging Face is awesome!", return_tensors="pt")
outputs = model(**inputs)

logits = outputs.logits
predicted_class = torch.argmax(logits, dim=-1).item()
print(f"Predicted class: {predicted_class}")

```

✔ **Runs inference using the fine-tuned model**

---

## **5️⃣ Managing Model Versions with Git**

💡 **Hugging Face models are stored as Git repositories.**

✅ **Clone Your Model Repository**

```bash
git clone <https://huggingface.co/username/bert-mrpc>
cd bert-mrpc

```

✔ **Creates a local copy of the repo**

✅ **Update & Push Changes**

```bash
git add .
git commit -m "Updated model"
git push

```

✔ **Syncs changes to the Hub**

✅ **Track Model Versions**

```bash
git log

```

✔ **Shows model version history**

---

## **6️⃣ Uploading Checkpoints & Logs**

💡 **Use `trainer.save_model()` to save & version intermediate checkpoints.**

✅ **Save Checkpoint Locally**

```python
trainer.save_model("checkpoint-1000")

```

✔ **Stores model weights at step 1000**

✅ **Push Checkpoint to Hub**

```python
!huggingface-cli upload bert-mrpc checkpoint-1000

```

✔ **Makes checkpoint accessible on the Hub**

✅ **Load Checkpoint Later**

```python
model = AutoModelForSequenceClassification.from_pretrained("username/bert-mrpc/checkpoint-1000")

```

✔ **Resumes training or runs inference on a specific checkpoint**

---

## **7️⃣ Running Model Inference via the Hub API**

💡 **The Hub provides an online Inference API for public models.**

✅ **Run API Inference**

```python
import requests

API_URL = "<https://api-inference.huggingface.co/models/username/bert-mrpc>"
headers = {"Authorization": "Bearer YOUR_HUGGINGFACE_TOKEN"}

data = {"inputs": "Hugging Face makes AI easy!"}
response = requests.post(API_URL, headers=headers, json=data)
print(response.json())

```

✔ **Runs inference without downloading the model**

✅ **Check Inference API on the Model Page**

1️⃣ **Go to `huggingface.co/{username}/bert-mrpc`**

2️⃣ **Enter text into the widget**

3️⃣ **See real-time predictions!** 🎯

---

## **8️⃣ Deleting & Managing Models**

💡 **Manage models via CLI or web interface.**

✅ **List Your Models**

```bash
huggingface-cli models

```

✔ **Shows all models in your account**

✅ **Delete a Model**

```bash
huggingface-cli delete bert-mrpc

```

✔ **Removes model from Hub**

✅ **Delete via Web Interface**

1️⃣ **Go to `huggingface.co/models`**

2️⃣ **Click on your model**

3️⃣ **Settings → Delete**

---

## **🎯 Summary – Key Takeaways**

✔ **Hugging Face Hub is a centralized repository for ML models**

✔ **Models, tokenizers, and logs can be uploaded via `push_to_hub()`**

✔ **Public models get an Inference API for free**

✔ **Versioning works via Git & CLI tools**

✔ **You can use models directly from the Hub with `from_pretrained()`**