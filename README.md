
# 🧠 Retrieval-Augmented Generation (RAG) with LangChain & Hugging Face

This project demonstrates how to build a simple **Retrieval-Augmented Generation (RAG)** system using the **LangChain** framework, **Hugging Face Transformers**, and **FAISS** for vector search. It loads a subset of the ArXiv Abstract-Title dataset and enables answering questions based on its content using a local LLM (`google/flan-t5-base`).

---

## 🚀 Features

- Uses LangChain for modular RAG pipeline construction
- Loads and chunks `.txt` documents from the ArXiv Abstract-Title dataset
- Embeds documents with `sentence-transformers/all-MiniLM-L6-v2`
- Stores vector embeddings in a FAISS vector store
- Uses Hugging Face `flan-t5-base` for generation (fully local)
- Supports question answering via `RetrievalQA` chain with `"stuff"` combination

---

## 📦 Setup Instructions

### 1. Install Dependencies

```bash
pip install langchain faiss-cpu transformers accelerate
pip install -U langchain-community
```

### 2. Download Dataset

```bash
wget -O arxiv_abs_title.zip "https://zenodo.org/records/3496527/files/gcunhase%2FArXivAbsTitleDataset-v1.0.zip?download=1"
unzip arxiv_abs_title.zip
```

### 3. Hugging Face Token (Optional)

If using hosted models (e.g., via `HuggingFaceHub`), you'll need a [Hugging Face API token](https://huggingface.co/settings/tokens). This notebook runs the model locally, so it's not required.

---

## 🛠️ How It Works

### Step 1: Load and Split Documents

- Loads two `.txt` files (abstract and title content)
- Splits them into overlapping text chunks for context retrieval

### Step 2: Generate Embeddings

- Uses `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) to convert chunks into dense vectors

### Step 3: Build a Vector Store

- Creates a FAISS index from the embeddings for fast similarity search

### Step 4: Setup Local LLM

- Loads `google/flan-t5-base` using Hugging Face Transformers' `pipeline`
- Wraps it with `HuggingFacePipeline` to integrate with LangChain

### Step 5: Construct RAG Pipeline

```python
from langchain.chains import RetrievalQA

rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)
```

- `chain_type="stuff"` means all retrieved documents are concatenated into a single prompt

### Step 6: Ask Questions

```python
question = "Can a modern email client proactively retrieve attachable items based on current conversation?"
response = rag_pipeline.run(question)
print(response)
```

---

## 📊 Chain Types

| Chain Type     | Description                                                              |
|----------------|--------------------------------------------------------------------------|
| `stuff`        | Simple, fast, combines all retrieved docs into a single prompt           |
| `map_reduce`   | Summarizes each doc then combines; better for long contexts              |
| `refine`       | Builds answer step-by-step; slower but more accurate                     |

---

## 📚 Dataset

- **Source**: [ArXiv Abstract-Title Dataset](https://zenodo.org/record/3496527)
- Contains paired abstracts and titles from scientific papers

---

## 📖 Reference

- Tutorial: [How to Build a Simple RAG System](https://medium.com/@mehar.chand.cloud/how-to-build-a-simple-retrieval-augmented-generation-rag-system-f6ffaf8a705c)
- LangChain: https://docs.langchain.com
- Hugging Face Transformers: https://huggingface.co/docs/transformers

---

## 📝 License

This project is provided for educational and research purposes only.
