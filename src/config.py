DATASET_NAME = "L-NLProc/PredEx_Instruction-Tuning_Pred-Exp"
COLUMNS_TO_KEEP = ['Case Name', 'Input', 'Output', 'Label', 'Count', 'Decision_Count', 'text']
MIN_WORDS = 50
MAX_WORDS = 6000

CHUNK_SIZE = 512
OVERLAP = 100

MODEL_NAME = "law-ai/InLegalBERT"
BATCH_SIZE = 128
UPSERT_BATCH = 100

PINECONE_API_KEY = "pcsk_2JYSAM_3hCKiJMHkUCtTPyQ9tM56fxA9Lf2RfPSuv4khzcXUJs3dFYvqk3Hp5YF1yhPJ3x"
INDEX_NAME = "legal-text-summarisation"
EMBEDDING_DIM = 768
METRIC = "cosine"

USE_MISTRAL = False
FLAN_MODEL_NAME = "google/flan-t5-large"
MISTRAL_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

LEGAL_SIGNAL_WORDS = [
    "held", "dismissed", "allowed", "rejected", "granted",
    "upheld", "quashed", "affirmed", "reversed", "remanded",
    "appeal is allowed", "appeal is dismissed",
    "petition is allowed", "petition is dismissed",
    "we agree", "we disagree", "we find", "we are of the opinion",
    "therefore", "accordingly", "consequently", "thus",
    "in view of", "in light of", "for the reasons",
    "it is clear", "it is evident", "it follows",
    "section", "article", "act", "statute", "provision",
    "this court", "high court", "supreme court",
    "respondent", "appellant", "petitioner"
]