### **📌 Mastering the 🤗 Datasets Library**  

---

## **1️⃣ What is 🤗 Datasets?**  

✔ **A library for loading, processing, and sharing datasets efficiently.**  
✔ **Built for NLP, but supports multiple data types.**  
✔ **Uses Apache Arrow for fast memory-mapped data access.**  

🎯 **Key Features:**  
✅ Load datasets from the Hugging Face Hub or local/remote sources.  
✅ Efficient dataset transformations with `Dataset.map()`.  
✅ Handle large datasets that don’t fit into RAM.  
✅ Compute metrics easily.  

💡 **Goal of this Chapter:**  
🚀 **Learn how to load and manipulate datasets like a pro!**  

---

## **2️⃣ Loading Datasets Not on the Hugging Face Hub**  

### **📌 1. Supported File Formats**  

🤗 Datasets supports multiple data formats:  

| **Data Format**  | **Loading Script**  | **Example** |
|----------------|------------------|-------------|
| **CSV & TSV**   | `csv`             | `load_dataset("csv", data_files="my_file.csv")` |
| **Text files**  | `text`            | `load_dataset("text", data_files="my_file.txt")` |
| **JSON & JSONL**| `json`            | `load_dataset("json", data_files="my_file.jsonl")` |
| **Pickled DataFrames** | `pandas` | `load_dataset("pandas", data_files="my_dataframe.pkl")` |

---

## **3️⃣ Loading a Local Dataset**  

📌 **Example: Loading SQuAD-it (Italian Question-Answering Dataset)**  

### **Step 1: Download & Extract Data**  
```bash
wget https://github.com/crux82/squad-it/raw/master/SQuAD_it-train.json.gz
wget https://github.com/crux82/squad-it/raw/master/SQuAD_it-test.json.gz
gzip -dkv SQuAD_it-*.json.gz  # Decompress the files
```

### **Step 2: Load the JSON Dataset**  
```python
from datasets import load_dataset

squad_it_dataset = load_dataset("json", data_files="SQuAD_it-train.json", field="data")
```

✔ **By default, it creates a `DatasetDict` with a `train` split.**  
✔ **To confirm, check:**  

```python
print(squad_it_dataset)
```
📌 **Output:**  
```python
DatasetDict({
    train: Dataset({
        features: ['title', 'paragraphs'],
        num_rows: 442
    })
})
```

### **Step 3: Access Dataset Elements**
```python
squad_it_dataset["train"][0]
```
📌 **Example Entry:**  
```python
{
    "title": "Terremoto del Sichuan del 2008",
    "paragraphs": [
        {
            "context": "Il terremoto del Sichuan del 2008 o il terremoto...",
            "qas": [
                {
                    "answers": [{"answer_start": 29, "text": "2008"}],
                    "id": "56cdca7862d2951400fa6826",
                    "question": "In quale anno si è verificato il terremoto nel Sichuan?",
                },
                ...
            ],
        },
        ...
    ],
}
```

---

## **4️⃣ Loading Multiple Splits in One DatasetDict**  

🚀 **To load both `train` and `test` splits together:**  

```python
data_files = {"train": "SQuAD_it-train.json", "test": "SQuAD_it-test.json"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
```

📌 **Output:**  
```python
DatasetDict({
    train: Dataset({
        features: ['title', 'paragraphs'],
        num_rows: 442
    })
    test: Dataset({
        features: ['title', 'paragraphs'],
        num_rows: 48
    })
})
```

---

## **5️⃣ Automatically Handling Compressed Files**  

🤗 Datasets can **load compressed files (GZIP, ZIP, TAR) directly** without manual decompression.  

✔ **Instead of decompressing manually, use:**  
```python
data_files = {"train": "SQuAD_it-train.json.gz", "test": "SQuAD_it-test.json.gz"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
```

🚀 **Saves time & avoids unnecessary storage usage!**  

---

## **6️⃣ Loading a Remote Dataset (From a URL)**  

💡 **Many datasets are stored on remote servers. Instead of downloading manually, load directly from the URL!**  

✔ **Example:**  
```python
url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
```

📌 **Same output as before, but saves you from manual downloads.**  

---

## **7️⃣ Advanced Data Loading Options**  

✔ **Load Multiple Files as One Split**  
```python
data_files = {"train": ["file1.json", "file2.json"]}
dataset = load_dataset("json", data_files=data_files, field="data")
```

✔ **Load Files Matching a Pattern (Using `glob`)**  
```python
dataset = load_dataset("json", data_files="data/*.json", field="data")
```

✔ **Load from Pandas DataFrame**  
```python
import pandas as pd
df = pd.read_csv("my_data.csv")
dataset = load_dataset("pandas", data_files="my_data.csv")
```

✔ **Load from Google Drive**  
```python
url = "https://drive.google.com/uc?id=<FILE_ID>"
dataset = load_dataset("json", data_files=url, field="data")
```

✔ **Load from Amazon S3**  
```python
s3_path = "s3://my-bucket/my_data.json"
dataset = load_dataset("json", data_files=s3_path, field="data")
```

---

## **🎯 Summary – Key Takeaways**  

🚀 **🤗 Datasets is powerful, efficient, and flexible!**  

✔ **Load datasets from local files, remote URLs, or even cloud storage.**  
✔ **Supports multiple formats (JSON, CSV, Text, Pandas, etc.).**  
✔ **Handles compressed files automatically.**  
✔ **Easily process & slice data with `Dataset.map()`.**  
✔ **Great for large-scale datasets (memory-efficient with Apache Arrow).**  

