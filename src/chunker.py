import pandas as pd
from tqdm import tqdm
from src.config import CHUNK_SIZE, OVERLAP

def chunking(tokenizer, input_text, case_name, label, explanation, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    tokenIds = tokenizer.encode(
        input_text,
        add_special_tokens=False,
        truncation=False,
    )

    chunks = []
    stride = chunk_size - overlap
    total_tokens = len(tokenIds)

    for s in range(0, total_tokens, stride):
        e = min(s + chunk_size, total_tokens)
        chunks_tokens = tokenIds[s:e]
        chunk_text = tokenizer.decode(chunks_tokens, skip_special_tokens=True)

        chunk_info = {
            'case_name': case_name,
            'chunk_idx': len(chunks),
            'chunk_text': chunk_text,
            'label': label,
            'explanation': explanation,
            'token_count': len(chunks_tokens),
            'is_last': (e == total_tokens)
        }
        chunks.append(chunk_info)
        if e == total_tokens:
            break

    return chunks

def chunk_dataframe(df, tokenizer, desc="Chunking"):
    all_chunks = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        doc_chunks = chunking(
            tokenizer=tokenizer,
            input_text=row['Input'],
            case_name=row['Case Name'],
            label=row['Label'],
            explanation=row['Explanation']
        )
        all_chunks.extend(doc_chunks)
    return pd.DataFrame(all_chunks)