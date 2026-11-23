# RAG with Unsloth Dynamic 4-bit Quantization

A retrieval-augmented question answering system built in one Google Colab notebook. It searches a small set of medical documents, hands the best passages to a 4-bit Llama model, and answers only from what it found. If the answer isn't in the documents, it says so.

This is Task 3 of my Generative AI internship.

## How it works

```
Question
  -> embed with all-MiniLM-L6-v2 (CPU)
  -> FAISS search, top 2 chunks
  -> grounded prompt (context + question + strict instructions)
  -> Llama-3.2-3B-Instruct, Unsloth dynamic 4-bit
  -> answer + source file names
```

| Part | Choice |
|---|---|
| Language model | `unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit` |
| Model loader | Unsloth `FastLanguageModel`, max sequence length 4096 |
| Framework | LangChain (loaders, text splitter, runnable chain) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, run on CPU |
| Vector store | FAISS (`faiss-cpu`) |
| Hardware | Colab Tesla T4 (14.56 GB) |

## Knowledge base

Five short educational documents on hypertension, type 2 diabetes, asthma, cold versus flu, and iron-deficiency anemia. I wrote them as plain text, and the notebook creates them in a `docs` folder when it runs. They are simplified summaries, not clinical guidelines.

The documents are split into chunks of 500 characters with a 50-character overlap, which gives 10 chunks in total. Malaria is deliberately left out so I can check that the system refuses questions outside its documents.

## What the notebook covers

- Loading the dynamic 4-bit model and checking VRAM.
- Building the documents, chunking them, embedding them and saving a FAISS index.
- A `rag_answer()` function that retrieves, builds a grounded prompt, generates, and prints the source files.
- Six test questions, including one that is not in the documents.
- A with-RAG against without-RAG comparison.
- The same pipeline written as a LangChain runnable chain.

## Results

| Measure | Value |
|---|---|
| Documents / chunks | 5 / 10 |
| VRAM after loading the model | 2.21 GB |
| Peak VRAM | 2.37 GB (2.41 GB reserved) |

The five in-domain questions were answered from the matching source file. The malaria question returned "I don't know based on the provided documents."

## Run it

1. Open `Intern_task3.ipynb` in Google Colab.
2. Set the runtime to **T4 GPU** (Runtime, then Change runtime type).
3. Run the cells from top to bottom. No files need to be uploaded.

## Limitations

- The knowledge base is tiny, so this shows the method rather than a usable medical tool.
- Retrieval is plain similarity search over 10 chunks, with no reranking.
- Answers use sampling at temperature 0.2, so wording can differ between runs.

## Disclaimer

This project is for learning only. It gives educational information, not medical advice.

## Author

Fairooz Nabilah Prova
