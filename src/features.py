import pandas as pd
from difflib import SequenceMatcher
import re

def string_sim(s1, s2):
    if pd.isna(s1) or pd.isna(s2):
        return 0.0
    return SequenceMatcher(None, str(s1), str(s2)).ratio()

def extract_numbers(s):
    if pd.isna(s):
        return set()
    return set(re.findall(r"\d+", str(s)))

def calculate_features(df_candidates, df_s1, df_s2_s3):
    df = df_candidates.merge(df_s1[["entity_id", "norm_name", "norm_address"]], left_on="source1_entity_id", right_on="entity_id")
    df = df.rename(columns={"norm_name": "s1_name", "norm_address": "s1_address"}).drop(columns=["entity_id"])
    
    df = df.merge(df_s2_s3[["entity_id", "norm_name", "norm_address"]], left_on="candidate_entity_id", right_on="entity_id")
    df = df.rename(columns={"norm_name": "s2_name", "norm_address": "s2_address"}).drop(columns=["entity_id"])
    
    # Text similarity features
    df["name_sim"] = df.apply(lambda x: string_sim(x["s1_name"], x["s2_name"]), axis=1)
    df["address_sim"] = df.apply(lambda x: string_sim(x["s1_address"], x["s2_address"]), axis=1)
    
    # Numerical address overlap (crucial for PIN codes and building numbers)
    def number_jaccard(row):
        n1 = extract_numbers(row["s1_address"])
        n2 = extract_numbers(row["s2_address"])
        if not n1 and not n2:
            return 1.0 # Both have no numbers -> agreement
        if not n1 or not n2:
            return 0.0
        return len(n1.intersection(n2)) / len(n1.union(n2))
        
    df["address_num_overlap"] = df.apply(number_jaccard, axis=1)
    
    # Drop string columns to save RAM
    return df.drop(columns=["s1_name", "s1_address", "s2_name", "s2_address"])
