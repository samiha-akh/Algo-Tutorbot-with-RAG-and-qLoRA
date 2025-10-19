# AlgoTutorbot

A tutoring system for algorithms and data structures, powered by fine-tuned LLMs with LoRA and Retrieval-Augmented Generation (RAG).

## Overview

AlgoTutor is a specialized AI assistant designed to help students learn algorithms and data structures. It combines:
- **Fine-tuned Mistral-7B** with QLoRA adapters trained on CS textbooks
- **RAG pipeline** with FAISS vector search for contextual retrieval
- **Interactive Gradio interface** for conversational learning

## Features

-  **Domain-Specific Knowledge**: Fine-tuned on multiple algorithms & data structures textbooks
-  **Contextual Responses**: Uses RAG to provide accurate, grounded answers
-  **Conversational Interface**: Natural dialogue with adjustable temperature settings
-  **Citation-Free**: Clean, direct explanations without reference IDs
-  **Efficient**: 4-bit quantization for resource-friendly deployment

##  Architecture

### 1. Data Processing Pipeline
```
PDFs → Text Extraction → Cleaning → Chunking → Master Dataset
```

### 2. Training Pipeline
```
Text Chunks → QA Generation (Gemma-2B) → Dataset Cleaning → QLoRA Fine-tuning
```

### 3. Inference Pipeline
```
User Query → Embedding → FAISS Retrieval → Context Augmentation → LLM Generation
```

## Prerequisites

```bash
Python 3.8+
CUDA-capable GPU (recommended)
16GB+ RAM
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/algotutor.git
cd algotutor
```

### 2. Install Dependencies
```bash
pip install -q transformers accelerate bitsandbytes peft trl datasets
pip install -q sentence-transformers faiss-cpu gradio
pip install -q pymupdf
```

### 3. Setup Hugging Face Authentication
```python
from huggingface_hub import login
login(token="your_hf_token_here")
```

## Dataset Preparation

### Step 1: Extract Text from PDFs

Run the data processor notebook to extract and clean text from CS textbooks:

```python
from algorizzm_dataprocessor import PDFProcessor

PDF_PATHS = [
    "path/to/algo-book1.pdf",
    "path/to/algo-book2.pdf",
    # Add more textbooks
]

processor = PDFProcessor(PDF_PATHS, output_dir="./data")
processor.run()
```

**Output**: `master_dataset.txt` (~4.75M characters)

### Step 2: Generate QA Pairs

Use Gemma-2B-IT to generate training data:

```python
python qa_generation.py
```

**Configuration**:
- Chunk size: 1500 characters
- Chunk overlap: 150 characters
- Batch size: 100 chunks
- Questions per chunk: 3

**Output**: `qa_finetuning_dataset.jsonl`

### Step 3: Clean Dataset

```python
python clean_dataset.py
```

Removes malformed JSON and validates Q&A structure.

##  Model Fine-Tuning

### QLoRA Configuration

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
)
```

### Training Parameters

```python
SFTConfig(
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    weight_decay=0.001,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03
)
```

### Run Training

```bash
python finetune_lora.py
```

**Base Model**: `mistralai/Mistral-7B-Instruct-v0.2`  
**Output**: LoRA adapters saved to `algo-tutor-mistral-7b-adapters/`

## RAG Setup

### Build FAISS Index

```python
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Load embedder
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Embed chunks
embeddings = embedder.encode(chunks, normalize_embeddings=True)

# Create FAISS index
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# Save
faiss.write_index(index, "faiss.index")
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)
```

## Usage

### Load Model with LoRA

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# Load base model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    quantization_config=bnb_config,
    device_map="auto"
)

# Attach LoRA adapters
model = PeftModel.from_pretrained(base_model, "path/to/adapters")
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
```

### Launch Gradio Interface

```python
python app.py
```

Or use the notebook interface for interactive experimentation.

### Example Queries

```
- "What is amortized analysis for dynamic arrays?"
- "Explain union-find with path compression and union by rank."
- "When is BFS different from Dijkstra's algorithm?"
- "How does quicksort work and what is its time complexity?"
```

## Configuration

### Temperature Control

Adjust creativity vs. accuracy:
- **Low (0.1-0.3)**: Focused, deterministic answers
- **Medium (0.4-0.7)**: Balanced explanations
- **High (0.8-1.0)**: More creative, diverse outputs

### Retrieval Settings

```python
K_INTERNAL = 10        # Number of chunks to retrieve
MAX_NEW_TOKS = 600     # Maximum response length
MIN_NEW_TOKS = 100     # Minimum response length
```

## Project Structure

```
algotutor/
├── algorizzm_dataprocessor.ipynb   # PDF extraction & cleaning
├── finetune_lora.ipynb             # QA generation & training
├── LLM_With_LoRA.ipynb             # Inference with RAG
├── data/
│   ├── master_dataset.txt          # Extracted textbook content
│   ├── qa_finetuning_dataset.jsonl # Training data
│   └── qa_finetuning_dataset_cleaned.jsonl
├── models/
│   └── algo-tutor-mistral-7b-adapters/  # LoRA weights
├── indices/
│   ├── faiss.index                 # Vector database
│   └── chunks.pkl                  # Text chunks
└── README.md
```

## Technical Details

### Model Quantization

Uses 4-bit NormalFloat (NF4) quantization with double quantization for memory efficiency:
- **Full Model**: ~14GB
- **Quantized**: ~4GB
- **Inference Speed**: ~10 tokens/sec on T4 GPU

### RAG Implementation

1. **Embedding**: MiniLM-L6-v2 (384 dimensions)
2. **Indexing**: FAISS IndexFlatIP (inner product similarity)
3. **Retrieval**: Top-10 chunks per query
4. **Context Window**: ~6000 tokens

### Prompt Template

```
<s>[INST] <<SYS>>
You are AlgoTutor, an algorithms & data-structures tutor.
Answer in detail USING ONLY the provided context passages.
If the answer is not in the context, say you don't know.
<</SYS>>

Question: {question}

Context:
{retrieved_passages}

Now answer using only the context above. [/INST]
```

## License

This project involves multiple licenses:

### Model Components

**Base Model (Mistral-7B-Instruct-v0.2)**
- Licensed under Apache 2.0
- See: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

**Training Data Generation (Gemma-2B-IT)**
- The Q&A training dataset was generated using Google's Gemma-2B-IT
- Subject to [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- **Important**: Models using Gemma-generated data must comply with Gemma's use restrictions
- Key restrictions include prohibited use policy (no harmful/illegal content, unlicensed professional advice, etc.)

### Use Restrictions

By using AlgoTutor or its derivatives, you agree to:
1. **Comply with Gemma Terms**: The training data was generated using Gemma, so downstream use must respect Google's prohibited use policy
2. **Educational Use**: This project is primarily intended for educational purposes
3. **No Harmful Use**: Do not use for illegal activities, generating harmful content, or providing unlicensed professional advice

### Third-Party Components
- **Sentence Transformers**: Apache 2.0
- **FAISS**: MIT License
- **Gradio**: Apache 2.0

**Note**: This project is for educational purposes. Ensure you have proper licenses for any textbooks used in training.
