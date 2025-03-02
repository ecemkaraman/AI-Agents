### **📌 Using Pretrained Models from the Hugging Face Hub**

---

## **1️⃣ Overview: Why Use Pretrained Models?**

💡 **The Hugging Face Hub** simplifies the process of:

✔ **Finding & selecting a model** for your task.

✔ **Loading models in just a few lines of code**.

✔ **Switching between different architectures seamlessly**.

✔ **Contributing & sharing models with the community**.

🚀 **Goal:** Learn how to:

🔹 Load and use a pretrained model.

🔹 Use the correct model for a given task.

🔹 Leverage `AutoModel` for flexibility.

---

## **2️⃣ Finding a Model for Your Task**

💡 **Example Task:** Masked Language Modeling (MLM) in **French**.

✅ **Selecting a Model:**

- We search for **French-based models** for **mask-filling**.
- We select **CamemBERT** (`camembert-base`).
- **Checkpoint Identifier:** `"camembert-base"`.

🔍 **Checkpoints are model-specific!** Using the wrong one **may lead to bad results**.

---

## **3️⃣ Using a Model with `pipeline()`**

💡 **Simplest way to use a model is via `pipeline()`.**

✅ **Load & Use the Model**

```python
from transformers import pipeline

# Load the fill-mask pipeline with Camembert
camembert_fill_mask = pipeline("fill-mask", model="camembert-base")

# Run inference
results = camembert_fill_mask("Le camembert est <mask> :)")
print(results)

```

✔ **Automatically downloads & loads the model**

✔ **Replaces `<mask>` with the most likely predictions**

✅ **Example Output**

```python
[
  {'sequence': 'Le camembert est délicieux :)', 'score': 0.49, 'token': 7200, 'token_str': 'délicieux'},
  {'sequence': 'Le camembert est excellent :)', 'score': 0.10, 'token': 2183, 'token_str': 'excellent'},
  {'sequence': 'Le camembert est succulent :)', 'score': 0.03, 'token': 26202, 'token_str': 'succulent'}
]

```

✔ **Top-3 Predictions**: `délicieux`, `excellent`, `succulent`.

---

## **4️⃣ Instantiating a Model Directly**

💡 **Alternative: Load model components separately**

✅ **Using Model-Specific Classes**

```python
from transformers import CamembertTokenizer, CamembertForMaskedLM

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForMaskedLM.from_pretrained("camembert-base")

```

✔ **Loads tokenizer & model separately**

✅ **Why Not Use This?**

❌ **Not flexible** – Only works with Camembert checkpoints.

❌ **Switching architectures requires changing code.**

---

## **5️⃣ Using AutoModel for Flexibility**

💡 **Best Practice: Use `Auto*` classes**

- **Why?** ✅ **AutoModel is architecture-agnostic**
- **Allows easy switching between models.**

✅ **Load Model with `Auto*` Classes**

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load tokenizer & model using Auto classes
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.from_pretrained("camembert-base")

```

✔ **Works with any MLM-compatible model**

✔ **Easy to switch models (just change checkpoint ID)**

✅ **Run Inference Manually**

```python
import torch

# Encode input
inputs = tokenizer("Le camembert est <mask> :)", return_tensors="pt")

# Get model predictions
outputs = model(**inputs)

# Extract predicted token
logits = outputs.logits
predicted_token_id = torch.argmax(logits, dim=-1)[0, -2].item()

# Decode output
predicted_word = tokenizer.decode(predicted_token_id)
print(f"Predicted word: {predicted_word}")

```

✔ **Manually extracts the masked word prediction**

---

## **6️⃣ Best Practices When Using Pretrained Models**

✅ **Check Model Cards**

📌 **Before using a model, check its training details:**

- **Datasets used**
- **Training limitations**
- **Potential biases**

✅ **Use `AutoModel` for Flexibility**

🚀 **Easier to swap models without rewriting code.**

✅ **Ensure Model Compatibility**

💡 **Make sure your model is suited for the task**

❌ **Example of an incorrect use case:**

```python
wrong_pipeline = pipeline("text-classification", model="camembert-base")
# ❌ Won’t work properly! Camembert is for MLM, not classification.

```

✔ **Use the task selector on the Hub** to find compatible models.

---

## **🎯 Summary – Key Takeaways**

✔ **Hugging Face Hub makes using pretrained models easy**

✔ **Use `pipeline()` for the simplest implementation**

✔ **Use `AutoTokenizer` & `AutoModel` for flexibility**

✔ **Check model documentation for training details & biases**

---

### **📌 Sharing Pretrained Models on the Hugging Face Hub**

---

## **1️⃣ Overview: Why Share Models?**

💡 **Sharing models helps the ML community by:**

✔ **Saving time & compute** – Others can use your model without retraining.

✔ **Providing reproducibility** – Share fine-tuned models for easy replication.

✔ **Contributing to open-source AI** – Build on top of existing work.

🚀 **Goal:** Learn how to upload models to the Hugging Face Hub using:

🔹 `push_to_hub()` API (Simple & automatic).

🔹 `huggingface_hub` Python library (More control).

🔹 Web interface (No code required).

🔹 Git & Git LFS (Manual but flexible).

---

## **2️⃣ Uploading with `push_to_hub()` API (Easiest Method)**

🔹 **Step 1: Authenticate to Hugging Face Hub**

```python
from huggingface_hub import notebook_login

notebook_login()

```

✔ **Alternatively, log in via CLI:**

```bash
huggingface-cli login

```

✔ **Enter your Hugging Face credentials.**

---

🔹 **Step 2: Upload Models During Training**

✅ If using `Trainer`, enable auto-upload:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    "bert-finetuned-mrpc",
    save_strategy="epoch",  # Save after every epoch
    push_to_hub=True        # Auto-upload to Hub
)

```

✔ **Trainer uploads after each epoch**

✔ **To upload final model:**

```python
trainer.push_to_hub()

```

✔ **Creates model card with metadata!**

---

🔹 **Step 3: Upload Pretrained Models Directly**

✅ Load a pretrained model:

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("camembert-base")
tokenizer = AutoTokenizer.from_pretrained("camembert-base")

```

✅ **Upload to Hub:**

```python
model.push_to_hub("dummy-model")
tokenizer.push_to_hub("dummy-model")

```

✔ **Model appears under your Hugging Face profile!**

✔ **To upload to an organization:**

```python
tokenizer.push_to_hub("dummy-model", organization="huggingface")

```

✔ **To specify an API token manually:**

```python
tokenizer.push_to_hub("dummy-model", use_auth_token="<TOKEN>")

```

🚀 **Now check your model on:**

🔗 `https://huggingface.co/user-or-organization/dummy-model`

---

## **3️⃣ Uploading with `huggingface_hub` Python Library (More Control)**

🔹 **Step 1: Install & Authenticate**

```bash
pip install huggingface_hub
huggingface-cli login

```

✔ **Same login method as `push_to_hub()`**

🔹 **Step 2: Create a Model Repository**

✅ Create a new repo:

```python
from huggingface_hub import create_repo

create_repo("dummy-model")  # Creates repo under your profile

```

✅ **To create under an organization:**

```python
create_repo("dummy-model", organization="huggingface")

```

✅ **To make the repo private:**

```python
create_repo("dummy-model", private=True)

```

---

🔹 **Step 3: Upload Model Files**

✅ **Direct upload of config files:**

```python
from huggingface_hub import upload_file

upload_file(
    "<path_to_file>/config.json",
    path_in_repo="config.json",
    repo_id="<namespace>/dummy-model",
)

```

✔ **Great for small files (<5GB).**

❌ **Not suitable for large model weights.**

---

## **4️⃣ Uploading via Web Interface (No Code Needed)**

✅ **Go to** [huggingface.co/new](https://huggingface.co/new)

✅ **Set owner, name, & visibility (public/private).**

✅ **After creation, manually upload files (README, model weights, etc.).**

✅ **Edit model card in Markdown format.**

🚀 **Easiest way for non-coders to share models!**

---

## **5️⃣ Uploading via Git & Git LFS (Manual but Flexible)**

🔹 **Step 1: Install Git LFS & Clone Repo**

```bash
git lfs install
git clone <https://huggingface.co/><namespace>/<model-name>
cd <model-name>

```

✔ **Creates a local repo for uploading files.**

🔹 **Step 2: Save Model & Tokenizer Locally**

```python
model.save_pretrained("<model-folder>")
tokenizer.save_pretrained("<model-folder>")

```

🔹 **Step 3: Add & Push Files to Hugging Face**

```bash
git add .
git commit -m "Uploading model files"
git push

```

✔ **Handles large files automatically via Git LFS.**

---

## **🎯 Summary – Key Takeaways**

✔ **Best for Beginners:** `push_to_hub()` (Fastest & simplest).

✔ **More Control:** `huggingface_hub` library (Repo management & file uploads).

✔ **No Code Needed:** Web interface (Great for beginners).

✔ **Power Users:** Git & Git LFS (Full control over repository).