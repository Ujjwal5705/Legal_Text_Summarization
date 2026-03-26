import time
import torch
import numpy as np
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from src.utils import id_generation
from src.config import BATCH_SIZE, UPSERT_BATCH, PINECONE_API_KEY, INDEX_NAME, EMBEDDING_DIM, METRIC

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def embed_text(text_list, tokenizer, model, device, desc="Embedding"):
    embeddings = []
    batch_iter = range(0, len(text_list), BATCH_SIZE)
    
    for i in tqdm(batch_iter, desc=desc, unit="batch"):
        batch = text_list[i:i + BATCH_SIZE]
        
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(device)

        with torch.no_grad():
            output = model(**encoded)

        cls_embeddings = output.last_hidden_state[:, 0, :]
        cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)
        embeddings.append(cls_embeddings.cpu().numpy())

        if (i // BATCH_SIZE) % 10 == 0 and device == "cuda":
            torch.cuda.empty_cache()

    return np.vstack(embeddings)

def setup_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric=METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(2)
            
    return pc.Index(INDEX_NAME)

def upsert_chunk_to_embeddings(df, tokenizer, model, device, index, namespace="train"):
    df = df.reset_index(drop=True)
    chunkText = df['chunk_text'].tolist()
    
    embeddings = embed_text(chunkText, tokenizer, model, device, desc=f"[{namespace}] Embedding")
    
    vectors = []
    for i, row in df.iterrows():
        vector_id = id_generation(row['case_name'], row['chunk_idx'])
        metadata = {
            "case_name": str(row['case_name'])[:100],
            "chunk_idx": int(row['chunk_idx']),
            "label": int(row['label']),
            "token_count": int(row['token_count']),
            "is_last": bool(row['is_last']),
            "chunk_preview": str(row['chunk_text'])[:500],
            "explanation_preview": str(row['explanation'])[:500]
        }
        df_idx = df.index.get_loc(i)
        emb_list = embeddings[df_idx].tolist()
        vectors.append((vector_id, emb_list, metadata))

    handles = []
    upserted = 0

    for start in tqdm(range(0, len(vectors), UPSERT_BATCH), desc="Upserting", unit="batch"):
        batch = vectors[start : start + UPSERT_BATCH]
        handle = index.upsert(vectors=batch, namespace=namespace, async_req=True)
        handles.append(handle)
        upserted += len(batch)

    for h in tqdm(handles, desc=f"[{namespace}] Confirming", unit="req"):
        h.get()

    return embeddings