import pandas as pd
import gc

def prep_keys(df):
    """
    Ultra-low memory extraction. Only returns the required keys.
    """
    keys_df = pd.DataFrame()
    keys_df["entity_id"] = df["entity_id"]
    keys_df["country"] = df["country"]
    
    name_clean = df["business_name"].fillna("").astype(str).str.lower().str.replace(r"[^\w\s]", "", regex=True).str.strip()
    name_splits = name_clean.str.split()
    keys_df["name_w1"] = name_splits.str[0].fillna("")
    keys_df["name_w2"] = name_splits.str[1].fillna("")
    del name_clean, name_splits
    
    addr_clean = df["business_address"].fillna("").astype(str).str.lower().str.replace(r"[^\w\s]", "", regex=True).str.strip()
    addr_splits = addr_clean.str.split()
    keys_df["addr_w1"] = addr_splits.str[0].fillna("")
    del addr_clean, addr_splits
    
    keys_df["zipcode"] = df["business_address"].fillna("").astype(str).str.extract(r"(\b\d{5,6}\b)")
    keys_df.fillna("", inplace=True)
    
    return keys_df

def generate_candidates(s1_keys, s23_keys):
    """
    Expects already prepped key dataframes to save memory.
    """
    def block(keys, max_freq=50):
        b1 = s1_keys[s1_keys[keys].notna().all(axis=1) & (s1_keys[keys] != "").all(axis=1)]
        b2 = s23_keys[s23_keys[keys].notna().all(axis=1) & (s23_keys[keys] != "").all(axis=1)]
        
        if len(b2) == 0 or len(b1) == 0:
            return pd.DataFrame(columns=["entity_id_x", "entity_id_y"])
            
        # PREVENT CARTESIAN EXPLOSION: frequency cap BOTH sides!
        # If a key combination appears more than 50 times in S1, it's a useless generic stop word. Drop it.
        key_counts_b1 = b1.groupby(keys).size()
        valid_keys_b1 = key_counts_b1[key_counts_b1 <= max_freq].index
        
        # If a key combination appears more than 50 times in the S2 chunk, drop it.
        key_counts_b2 = b2.groupby(keys).size()
        valid_keys_b2 = key_counts_b2[key_counts_b2 <= max_freq].index
        
        # Keep only keys that are valid in BOTH sides to be safe
        valid_keys = valid_keys_b1.intersection(valid_keys_b2)
        
        if len(valid_keys) == 0:
            return pd.DataFrame(columns=["entity_id_x", "entity_id_y"])
        
        if len(keys) == 1:
            b1 = b1[b1[keys[0]].isin(valid_keys)]
            b2 = b2[b2[keys[0]].isin(valid_keys)]
        else:
            b1 = b1.set_index(keys).loc[b1.set_index(keys).index.isin(valid_keys)].reset_index()
            b2 = b2.set_index(keys).loc[b2.set_index(keys).index.isin(valid_keys)].reset_index()
            
        merged = pd.merge(b1[["entity_id"] + keys], b2[["entity_id"] + keys], on=keys)
        return merged[["entity_id_x", "entity_id_y"]].rename(columns={"entity_id_x": "source1_entity_id", "entity_id_y": "candidate_entity_id"})

    p1 = block(["country", "name_w1", "addr_w1"])
    p2 = block(["country", "name_w1", "name_w2"])
    p3 = block(["country", "name_w1", "zipcode"])
    
    if len(p1) == 0 and len(p2) == 0 and len(p3) == 0:
        candidates = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
    else:
        candidates = pd.concat([p1, p2, p3], ignore_index=True).drop_duplicates()
        
    candidates["blocking_score"] = 1.0 
    
    del p1, p2, p3
    gc.collect()
    
    return candidates
