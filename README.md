# Legal Text Summarization & Outcome Prediction (RAG Pipeline)

This repository contains a modular Retrieval-Augmented Generation (RAG) pipeline designed for Indian legal text. It predicts the outcome of legal cases (Accepted/Rejected) and generates an extractive/abstractive legal summary explaining the reasoning, based on the `L-NLProc/PredEx_Instruction-Tuning_Pred-Exp` dataset.

## Features
* **Data Processing:** Automated downloading, cleaning, and filtering of legal text datasets.
* **Intelligent Chunking:** Overlapping text chunking using `law-ai/InLegalBERT`.
* **Vector Database Integration:** Automated generation of text embeddings and asynchronous upserting to **Pinecone** for fast semantic retrieval.
* **Custom Sentence Scoring:** Ranks the most important sentences in a case using a weighted formula based on document centroid similarity, position bias (heavier weight to the end of documents), and legal keyword frequency.
* **LLM Generation:** Uses `google/flan-t5-large` (with support for `Mistral-7B`) to process retrieved precedents and predict case outcomes.
* **Automated Evaluation:** Built-in calculation of ROUGE-L and BERTScore metrics to evaluate the generated summaries against ground-truth expert explanations.

## Project Structure

<img width="498" height="530" alt="image" src="https://github.com/user-attachments/assets/4c7645b4-9290-46c4-9955-a066c449dc43" />


## Installation & Setup

1. **Clone the repository:**
   ```
   git clone https://github.com/yourusername/legal-text-summarization.git
   cd legal-text-summarization
   ```

2. **Install dependencies:**
   Ensure you have Python 3.8+ installed, then run:
   ```
   pip install -r requirements.txt
   ```
   *(Note: If you plan to switch to the Mistral model in `config.py`, you will also need to install `bitsandbytes` and `accelerate` for 4-bit quantization).*

3. **Configure Pinecone:**
   Open `src/config.py` and ensure your `PINECONE_API_KEY` is correct. The pipeline will automatically create the required index (`legal-text-summarisation`) if it doesn't already exist.

## Usage

To run the entire pipeline end-to-end, simply execute:

```
python main.py
```

**What this does:**
1. Downloads and cleans the dataset.
2. Chunks the text and generates embeddings using `InLegalBERT`.
3. Pushes the embeddings to Pinecone.
4. Scores individual sentences in the cases to find the most important facts.
5. Queries the LLM (`Flan-T5` by default) using the RAG augmented prompt.
6. Evaluates the outputs and saves the final predictions and metrics to `pipeline_results.csv`.

## Configuration 
You can tweak the pipeline's behavior by editing `src/config.py`:
* **`CHUNK_SIZE` / `OVERLAP`:** Adjust the token limits for chunking.
* **`USE_MISTRAL`:** Set to `True` to switch from Flan-T5 to Mistral-7B for LLM generation.
* **`BATCH_SIZE` / `UPSERT_BATCH`:** Adjust based on your GPU VRAM and network limits.
