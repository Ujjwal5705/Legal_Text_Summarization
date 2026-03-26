import pandas as pd
from datasets import load_dataset
from src.utils import cleantext, split_output, drop_empty_columns, filter_by_length
from src.config import DATASET_NAME, COLUMNS_TO_KEEP, MIN_WORDS, MAX_WORDS

def load_and_preprocess_data():
    train_dataset = load_dataset(DATASET_NAME, split="train")
    test_dataset = load_dataset(DATASET_NAME, split="test")
    
    train_df = train_dataset.to_pandas()
    test_df = test_dataset.to_pandas()
    
    train_df = drop_empty_columns(train_df, COLUMNS_TO_KEEP)
    test_df = drop_empty_columns(test_df, COLUMNS_TO_KEEP)
    
    train_df = split_output(train_df)
    test_df = split_output(test_df)
    
    for df in [train_df, test_df]:
        df['Case Name'] = df['Case Name'].apply(cleantext)
        df['Explanation'] = df['Explanation'].apply(cleantext)
        df['Input'] = df['Input'].apply(cleantext)
        
    print("Train:")
    train_df = filter_by_length(train_df, MIN_WORDS, MAX_WORDS)
    print("Test:")
    test_df = filter_by_length(test_df, MIN_WORDS, MAX_WORDS)
    
    return train_df, test_df