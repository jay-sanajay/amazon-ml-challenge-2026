import pandas as pd
import gc

def generate_candidates(df_s1, df_s2_s3):
    """
    Dictionary blocking with explicit Cartesian explosion prevention.
    Limits the maximum frequency of block keys to prevent OOM.
    """
    def prep_keys(df):
        df = df.copy()
        name_clean = df["business_name"].str.lower().str.replace(r"[^\w\s]", "", regex=True).str.strip()
        addr_clean = df["business_address"].str.lower().str.replace(r"[^\w\s]", "", regex=True).str.strip()
        
        name_splits = name_clean.str.split()
        df["name_w1"] = name_splits.str[0].fillna("")
        df["name_w2"] = name_splits.str[1].fillna("")
        
        addr_splits = addr_clean.str.split()
        df["addr_w1"] = addr_splits.str[0].fillna("")
        
        df["zipcode"] = df["business_address"].str.extract(r"(\b\d{5,6}\b)")
        df["zipcode"] = df["zipcode"].fillna("")
        return df

    s1 = prep_keys(df_s1)
    s23 = prep_keys(df_s2_s3)
    
    def block(keys, max_freq=50):
        b1 = s1[s1[keys].notna().all(axis=1) & (s1[keys] != "").all(axis=1)]
        b2 = s23[s23[keys].notna().all(axis=1) & (s23[keys] != "").all(axis=1)]
        
        # PREVENT CARTESIAN EXPLOSION: 
        # Count frequency of keys in B2 and drop overly common ones (e.g., generic words like "the")
        key_counts = b2.groupby(keys).size()
        valid_keys = key_counts[key_counts <= max_freq].index
        
        if len(keys) == 1:
            b2 = b2[b2[keys[0]].isin(valid_keys)]
        else:
            b2 = b2.set_index(keys).loc[b2.set_index(keys).index.isin(valid_keys)].reset_index()
            
        merged = pd.merge(b1[["entity_id"] + keys], b2[["entity_id"] + keys], on=keys)
        return merged[["entity_id_x", "entity_id_y"]].rename(columns={"entity_id_x": "source1_entity_id", "entity_id_y": "candidate_entity_id"})

    p1 = block(["country", "name_w1", "addr_w1"])
    p2 = block(["country", "name_w1", "name_w2"])
    p3 = block(["country", "name_w1", "zipcode"])
    
    candidates = pd.concat([p1, p2, p3]).drop_duplicates()
    candidates["blocking_score"] = 1.0 
    
    del s1, s23, p1, p2, p3
    gc.collect()
    
    return candidates
