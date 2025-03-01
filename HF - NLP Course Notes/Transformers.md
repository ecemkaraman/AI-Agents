### **🚀 Transformer Models & Hugging Face Pipelines**

---

## **1️⃣ Introduction to Transformer Models**

- **Used for various NLP tasks** → Sentiment analysis, translation, summarization, etc.
- **Hugging Face Model Hub** → Thousands of pretrained models available
- **Supports both cloud (Colab) & local execution**

---

## **2️⃣ Working with Pipelines**

💡 **pipeline()** → Automates NLP tasks by handling **preprocessing, model inference, and post-processing**.

### **▶️ Example: Sentiment Analysis**

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")
```

**🔹 Output:** `[{'label': 'POSITIVE', 'score': 0.96}]`

📌 **Key Features:**

- Accepts multiple sentences at once
- Uses a **cached model** (no need to redownload)

---

## **3️⃣ NLP Tasks & Pipelines**

💡 **Common Pipelines** in 🤗 Transformers:

- **feature-extraction** → Vector representation of text
- **fill-mask** → Predict missing words
- **ner** → Named Entity Recognition (NER)
- **question-answering** → Extract answers from text
- **sentiment-analysis** → Classify emotions
- **summarization** → Shorten long text
- **text-generation** → Predict next words in sequence
- **translation** → Convert text between languages
- **zero-shot-classification** → Classify text without labeled training data

---

## **4️⃣ Zero-Shot Classification**

- No fine-tuning required
- Classifies text **with custom labels**

### **▶️ Example: Custom Label Classification**

```python
classifier = pipeline("zero-shot-classification")
classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"]
)
```

**🔹 Output:**

```json
{'labels': ['education', 'business', 'politics'], 'scores': [0.84, 0.11, 0.04]}

```

📌 **Why "Zero-Shot"?** → Works with **any label set** without retraining!

---

## **5️⃣ Text Generation**

- Generates text based on a **prompt**
- Similar to predictive text on smartphones

### **▶️ Example: Basic Text Generation**

```python
generator = pipeline("text-generation")
generator("In this course, we will teach you how to")
```

📌 **Customizable with:**

- `num_return_sequences` → Number of variations
- `max_length` → Limit output length

---

## **6️⃣ Using Custom Models from the Hub**

💡 Instead of using default models, specify **custom models** from Hugging Face Hub.

### **▶️ Example: Using `distilgpt2` for Text Generation**

```python
generator = pipeline("text-generation", model="distilgpt2")
generator("In this course, we will teach you how to", max_length=30, num_return_sequences=2)
```

📌 **Find models by:**

- Searching in **Model Hub**
- Filtering by **language & task**

---

## **7️⃣ Mask Filling (Fill-in-the-Blanks)**

- Predicts **missing words** in a sentence

### **▶️ Example: Fill-Mask Pipeline**

```python
unmasker = pipeline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=2)

```

🔹 *Example Predictions:*

```json
[{'sequence': 'This course will teach you all about mathematical models.'},
 {'sequence': 'This course will teach you all about computational models.'}]

```

📌 **Check model-specific `<mask>` token** before using!

---

## **8️⃣ Named Entity Recognition (NER)**

- **Identifies persons, locations, organizations, etc.**

### **▶️ Example: NER Pipeline**

```python
ner = pipeline("ner", grouped_entities=True)
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")

```

🔹 *Output:*

```json
[{'entity_group': 'PER', 'word': 'Sylvain'},
 {'entity_group': 'ORG', 'word': 'Hugging Face'},
 {'entity_group': 'LOC', 'word': 'Brooklyn'}]

```

📌 **`grouped_entities=True`** → Merges multi-word entities.

---

## **9️⃣ Question Answering**

- Extracts **answers** from a **given context**

### **▶️ Example: QA Pipeline**

```python
question_answerer = pipeline("question-answering")
question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn."
)

```

🔹 *Output:*

```json
{'answer': 'Hugging Face'}

```

📌 **Uses context** to find **existing answers** (doesn’t generate new ones).

---

## **🔟 Summarization**

- **Condenses long text** into key points

### **▶️ Example: Summarization Pipeline**

```python
summarizer = pipeline("summarization")
summarizer("Long text here...")

```

📌 **Customizable with:**

- `max_length` → Shorten output
- `min_length` → Retain key details

---

## **1️⃣1️⃣ Translation**

- Converts **text between languages**

### **▶️ Example: French → English Translation**

```python
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face.")

```

🔹 *Output:* `'This course is produced by Hugging Face.'`

📌 **Find models for different languages** in **Model Hub**

---

## **1️⃣2️⃣ Inference API**

🔗 **Test models online via Hugging Face's web interface**

- Allows **quick experimentation** before downloading
- Available as a **paid API** for integration into workflows

---

### **🎯 Next Steps**

📌 Pipelines = **Prebuilt solutions** for common NLP tasks

📌 **Next chapter:** 🔍 *Deep dive into pipeline internals* & **customization** 🚀