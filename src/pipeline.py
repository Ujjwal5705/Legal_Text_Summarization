import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from src.utils import parse_prediction
from src.config import USE_MISTRAL

def get_top_k_sentences(case_name, scored_df, top_k=5):
    case_sents = scored_df[scored_df['case_name'] == case_name].copy()
    if case_sents.empty:
        return ""
    top_sents = case_sents.nlargest(top_k, 'score')
    top_sents = top_sents.sort_values('position')
    return " ".join(top_sents['sentence'].tolist())

def embed_single(text, tokenizer, model, device):
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        out = model(**encoded)

    cls = out.last_hidden_state[:, 0, :]
    cls = torch.nn.functional.normalize(cls, p=2, dim=1)
    return cls.cpu().numpy()[0].tolist()

def rag_retrieve(query_text, index, tokenizer, model, device, top_k=3, namespace="train"):
    query_embedding = embed_single(query_text[:2000], tokenizer, model, device)
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )

    retrieved = []
    seen_cases = set()

    for match in results['matches']:
        meta = match['metadata']
        case_name = meta.get('case_name', '')
        if case_name in seen_cases:
            continue
        seen_cases.add(case_name)

        retrieved.append({
            'case_name': case_name,
            'label': meta.get('label', -1),
            'chunk_preview': meta.get('chunk_preview', ''),
            'explanation_preview': meta.get('explanation_preview', ''),
            'similarity_score': round(match['score'], 4)
        })

    return retrieved

def build_augmented_prompt(query_case_text, top_k_summary, retrieved_cases, model_type="flan-t5"):
    if model_type == "flan-t5":
        similar_context = ""
        for i, case in enumerate(retrieved_cases[:2]):
            lbl = "accepted" if case['label'] == 1 else "rejected"
            similar_context += f"Similar case {i+1} (outcome: {lbl}):\n{case['chunk_preview'][:200]}\n\n"

        return (
            f"Task: Predict the outcome and explain the legal reasoning.\n\n"
            f"Key facts from current case:\n{top_k_summary[:400]}\n\n"
            f"Similar past cases for reference:\n{similar_context}"
            f"Predict outcome (0=rejected, 1=accepted) and explain:\n"
        )
    else:
        similar_context = ""
        for i, case in enumerate(retrieved_cases[:3]):
            lbl = "accepted" if case['label'] == 1 else "rejected"
            similar_context += (
                f"### Similar Case {i+1} (Outcome: {lbl})\n"
                f"{case['chunk_preview'][:400]}\n"
                f"Reasoning: {case['explanation_preview'][:300]}\n\n"
            )

        return (
            f"### Instructions\n"
            f"You are an expert Indian legal AI assistant. Given the case details and similar precedents below, "
            f"predict the outcome (0=rejected, 1=accepted) and provide a detailed legal explanation.\n\n"
            f"### Key Facts from Current Case\n{top_k_summary[:800]}\n\n"
            f"### Similar Precedents\n{similar_context}"
            f"### Full Case Text (truncated)\n{query_case_text[:1500]}\n\n"
            f"### Response (format: Prediction: [0 or 1]\nExplanation: [reasoning])\n"
        )

def evaluate_single(generated, reference, pred_label, true_label):
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rl = rouge.score(reference, generated)['rougeL']
    return {
        'rouge_l_f': round(rl.fmeasure, 4),
        'rouge_l_p': round(rl.precision, 4),
        'rouge_l_r': round(rl.recall, 4),
        'pred_label': pred_label,
        'true_label': true_label,
        'label_correct': int(pred_label == true_label)
    }

def evaluate_batch_bertscore(generated_list, reference_list, device):
    P, R, F1 = bert_score(
        generated_list,
        reference_list,
        lang="en",
        device=device,
        batch_size=16,
        verbose=False
    )
    return F1.tolist()

def run_full_pipeline(test_df, scored_df, index, emb_tokenizer, emb_model, llm_tokenizer, llm_model, device, generate_fn, top_k_sentences=5, rag_top_k=3, max_cases=None):
    if max_cases:
        test_df = test_df.head(max_cases)

    results = []
    generated_list = []
    reference_list = []
    model_type = "mistral" if USE_MISTRAL else "flan-t5"

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Running pipeline"):
        case_name = str(row.get('Case Name', row.get('case_name', '')))
        input_text = str(row['Input'])
        true_label = int(row['Label'])
        reference = str(row['Explanation'])

        top_k_text = get_top_k_sentences(case_name, scored_df, top_k=top_k_sentences)
        if not top_k_text:
            top_k_text = input_text[:500]

        retrieved = rag_retrieve(input_text, index, emb_tokenizer, emb_model, device, top_k=rag_top_k, namespace="train")

        prompt = build_augmented_prompt(
            query_case_text=input_text,
            top_k_summary=top_k_text,
            retrieved_cases=retrieved,
            model_type=model_type
        )

        llm_output = generate_fn(prompt, llm_tokenizer, llm_model, device)
        pred_label = parse_prediction(llm_output)

        metrics = evaluate_single(llm_output, reference, pred_label, true_label)

        results.append({
            'case_name': case_name,
            'true_label': true_label,
            'pred_label': pred_label,
            'rag_retrieved': [c['case_name'] for c in retrieved],
            'rag_scores': [c['similarity_score'] for c in retrieved],
            'llm_output': llm_output,
            'reference': reference,
            **metrics
        })

        generated_list.append(llm_output)
        reference_list.append(reference)

    results_df = pd.DataFrame(results)
    bert_f1_scores = evaluate_batch_bertscore(generated_list, reference_list, device)
    results_df['bertscore_f1'] = bert_f1_scores

    return results_df