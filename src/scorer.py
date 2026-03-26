import numpy as np
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from src.config import LEGAL_SIGNAL_WORDS

def get_embeddings_for_sentences(texts, tokenizer, model, device, batch_size=64):
    res = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            out = model(**encoded)
            
        cls = out.last_hidden_state[:, 0, :]
        cls = torch.nn.functional.normalize(cls, p=2, dim=1)
        res.append(cls.cpu().numpy())
    return np.vstack(res)

def score_sentence(doc_text, tokenizer, model, device, top_k=10):
    sentences = sent_tokenize(doc_text)
    sentences = [s.strip() for s in sentences if len(s.split()) >= 5]

    if len(sentences) == 0:
        return []
        
    n = len(sentences)
    all_texts = sentences + [doc_text[:1000]]
    all_embs = get_embeddings_for_sentences(all_texts, tokenizer, model, device)
    
    sent_embs = all_embs[:-1]
    doc_centroid = all_embs[-1:]

    sim_scores = (sent_embs @ doc_centroid.T).flatten()
    sim_scores = (sim_scores - sim_scores.min()) / (sim_scores.max() - sim_scores.min() + 1e-8)

    positions = np.arange(n)
    pos_scores = positions / (n - 1) if n > 1 else np.ones(n)
    last_30_pct = int(0.7 * n)
    pos_boost = np.where(positions >= last_30_pct, 1.0, pos_scores)
    pos_scores = pos_boost

    kw_scores = np.zeros(n)
    for idx, sent in enumerate(sentences):
        sent_lower = sent.lower()
        hits = sum(1 for kw in LEGAL_SIGNAL_WORDS if kw in sent_lower)
        kw_scores[idx] = min(hits / 3.0, 1.0)

    final_scores = (0.60 * sim_scores + 0.25 * pos_scores + 0.15 * kw_scores)
    ranked_indices = np.argsort(final_scores)[::-1][:top_k]
    top_indices = sorted(ranked_indices.tolist())

    results = []
    for idx in top_indices:
        results.append({
            "sentence": sentences[idx],
            "position": idx,
            "score": round(float(final_scores[idx]), 4),
            "sim_score": round(float(sim_scores[idx]), 4),
            "pos_score": round(float(pos_scores[idx]), 4),
            "kw_score": round(float(kw_scores[idx]), 4),
        })

    return results

def score_dataset(df, tokenizer, model, device, top_k=10, desc="Scoring"):
    all_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        case_name = str(row.get('Case Name', row.get('case_name', 'unknown')))
        label = int(row['Label'])
        explanation = str(row['Explanation'])
        doc_text = str(row['Input'])

        scored = score_sentence(doc_text, tokenizer, model, device, top_k=top_k)

        for sent_info in scored:
            all_rows.append({
                "case_name": case_name,
                "label": label,
                "explanation": explanation,
                **sent_info
            })

    return pd.DataFrame(all_rows)