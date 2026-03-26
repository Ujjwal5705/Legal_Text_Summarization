import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration, T5Tokenizer
import nltk

from src.data_loader import load_and_preprocess_data
from src.chunker import chunk_dataframe
from src.embedder import get_device, setup_pinecone, upsert_chunk_to_embeddings
from src.scorer import score_dataset
from src.pipeline import run_full_pipeline
from src.config import MODEL_NAME, FLAN_MODEL_NAME

def generate_summary_flan(prompt, tokenizer, model, device):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    
    device = get_device()
    print(f"Running on: {device}")
    
    print("Loading Base Models...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    train_df, test_df = load_and_preprocess_data()
    
    print("Chunking Data...")
    train_chunks = chunk_dataframe(train_df, tokenizer, desc="Train Chunking")
    test_chunks = chunk_dataframe(test_df, tokenizer, desc="Test Chunking")
    
    print("Connecting to Pinecone...")
    index = setup_pinecone()
    
    print("Upserting to Pinecone...")
    upsert_chunk_to_embeddings(train_chunks, tokenizer, model, device, index, namespace="train")
    upsert_chunk_to_embeddings(test_chunks, tokenizer, model, device, index, namespace="test")

    print("Scoring Sentences...")
    train_scored = score_dataset(train_df, tokenizer, model, device, top_k=10, desc="Score Train")
    test_scored = score_dataset(test_df, tokenizer, model, device, top_k=10, desc="Score Test")
    
    print("Loading LLM for generation...")
    llm_tokenizer = T5Tokenizer.from_pretrained(FLAN_MODEL_NAME)
    llm_model = T5ForConditionalGeneration.from_pretrained(FLAN_MODEL_NAME, torch_dtype=torch.float16).to(device)
    llm_model.eval()
    
    print("Running Full RAG Pipeline...")
    results_df = run_full_pipeline(
        test_df=test_df,
        scored_df=test_scored,
        index=index,
        emb_tokenizer=tokenizer,
        emb_model=model,
        llm_tokenizer=llm_tokenizer,
        llm_model=llm_model,
        device=device,
        generate_fn=generate_summary_flan,
        max_cases=50 
    )
    
    results_df.to_csv("pipeline_results.csv", index=False)
    print("Pipeline complete. Results saved to pipeline_results.csv")

if __name__ == "__main__":
    main()