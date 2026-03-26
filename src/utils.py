import re

def id_generation(name, chunk_idx):
    clean_name = ""
    for c in name[:40]:
        if c.isalnum():
            clean_name += c
        else:
            clean_name += "_"
    return f"{clean_name}_chunk_{chunk_idx}"

def parse_prediction(llm_output):
    text = llm_output.lower()
    
    m = re.search(r'prediction[:\s]+([01])', text)
    if m:
        return int(m.group(1))

    if any(w in text for w in ['allowed', 'accepted', 'granted', 'upheld']):
        return 1
    if any(w in text for w in ['dismissed', 'rejected', 'denied', 'quashed']):
        return 0

    m = re.search(r'\b([01])\b', text)
    if m:
        return int(m.group(1))

    return -1