import pandas as pd
import os
import gc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../dataset")
OUT_DIR = os.path.join(BASE_DIR, "../../output")
os.makedirs(OUT_DIR, exist_ok=True)

def clean_data(df):
    df["name_clean"] = df["business_name"].fillna("").astype(str).str.lower().str.replace(r"[^\w\s]", "", regex=True)
    df["addr_clean"] = df["business_address"].fillna("").astype(str).str.lower().str.replace(r"[^\w\s]", "", regex=True)
    
    name_splits = df["name_clean"].str.split()
    df["name_w1"] = name_splits.str[0].fillna("")
    df["name_w2"] = name_splits.str[1].fillna("")
    
    df["zipcode"] = df["business_address"].fillna("").astype(str).str.extract(r"(\b\d{5,6}\b)")
    df["zipcode"] = df["zipcode"].fillna("")
    
    df["country"] = df["country"].fillna("UNKNOWN")
    df["addr_w1"] = df["addr_clean"].str.split().str[0].fillna("")
    return df[["entity_id", "country", "name_w1", "name_w2", "zipcode", "name_clean", "addr_clean", "addr_w1"]]

def process_chunk(chunk_path, s1_df, all_matches):
    print(f"Processing {os.path.basename(chunk_path)}...")
    for chunk in pd.read_csv(chunk_path, sep="\t", dtype=str, usecols=["entity_id", "business_name", "business_address", "country"], chunksize=100000):
        c_df = clean_data(chunk)
        del chunk
        gc.collect()
        
        # Candidate Generation Strategy: Union of multiple loose blocks to maximize recall
        def block(keys, max_freq=50):
            b1 = s1_df[s1_df[keys].notna().all(axis=1) & (s1_df[keys] != "").all(axis=1)]
            b2 = c_df[c_df[keys].notna().all(axis=1) & (c_df[keys] != "").all(axis=1)]
            
            if len(b2) == 0 or len(b1) == 0:
                return pd.DataFrame()
                
            k1 = b1.groupby(keys).size().reset_index(name='c')
            v1 = k1[k1['c'] <= max_freq].drop(columns=['c'])
            k2 = b2.groupby(keys).size().reset_index(name='c')
            v2 = k2[k2['c'] <= max_freq].drop(columns=['c'])
            vk = pd.merge(v1, v2, on=keys, how="inner")
            
            if len(vk) > 0:
                m1 = pd.merge(b1, vk, on=keys, how="inner")
                m2 = pd.merge(b2, vk, on=keys, how="inner")
                cols1 = list(set(["entity_id", "country", "name_clean", "addr_clean"] + keys))
                cols2 = list(set(["entity_id", "country", "name_clean", "addr_clean"] + keys))
                return pd.merge(m1[cols1], m2[cols2], on=keys)
            return pd.DataFrame()

        p1 = block(["country", "name_w1", "name_w2"], max_freq=500)
        p2 = block(["country", "name_w1", "zipcode"], max_freq=500)
        p3 = block(["country", "name_clean"], max_freq=2000)
        
        # Catch businesses with name typos but exact address matches
        p4 = block(["country", "addr_w1", "zipcode"], max_freq=500)
        
        cands_list = [p for p in [p1, p2, p3, p4] if not p.empty]
        if not cands_list:
            del c_df
            gc.collect()
            continue
            
        cands = pd.concat(cands_list, ignore_index=True).drop_duplicates(subset=["entity_id_x", "entity_id_y"])
        
        # Highly accurate fuzzy string matching
        from difflib import SequenceMatcher
        def fast_sim(list1, list2):
            return [SequenceMatcher(None, a, b).quick_ratio() for a, b in zip(list1, list2)]
            
        cands["name_score"] = fast_sim(cands["name_clean_x"], cands["name_clean_y"])
        cands["addr_score"] = fast_sim(cands["addr_clean_x"], cands["addr_clean_y"])
        
        # To hit 100% accuracy, demand exceptionally high spelling similarity
        strict_match = cands[
            (cands["name_score"] >= 0.90) | 
            ((cands["name_score"] >= 0.75) & (cands["addr_score"] >= 0.75))
        ]
        
        if len(strict_match) > 0:
            all_matches.append(strict_match[["entity_id_x", "entity_id_y"]].rename(columns={"entity_id_x": "source1_entity_id", "entity_id_y": "candidate_entity_id"}))
            
        del p1, p2, cands_list, cands, c_df
        gc.collect()

def main():
    print("--- STARTING OPTIMIZED F0.5 PIPELINE ---")
    s1_path = os.path.join(DATA_DIR, "test/test_source1.tsv")
    s2_path = os.path.join(DATA_DIR, "test/test_source2.tsv")
    s3_path = os.path.join(DATA_DIR, "test/test_source3.tsv")
    
    s1_accum = []
    for chunk in pd.read_csv(s1_path, sep="\t", dtype=str, usecols=["entity_id", "business_name", "business_address", "country"], chunksize=100000):
        s1_accum.append(clean_data(chunk))
    s1 = pd.concat(s1_accum, ignore_index=True)
    del s1_accum
    gc.collect()
    
    all_matches = []
    
    process_chunk(s2_path, s1, all_matches)
    process_chunk(s3_path, s1, all_matches)
    
    print("Saving outputs...")
    if all_matches:
        final_cands = pd.concat(all_matches, ignore_index=True).drop_duplicates()
    else:
        final_cands = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
        
    cand_out = final_cands.groupby("source1_entity_id")["candidate_entity_id"].apply(lambda x: ",".join(x)).reset_index()
    cand_out.columns = ["source1_entity_id", "matched_entity_ids"]
    
    s1_ids = pd.read_csv(s1_path, sep="\t", dtype=str, usecols=["entity_id"]).rename(columns={"entity_id": "source1_entity_id"})
    
    cand_final = s1_ids.merge(cand_out, on="source1_entity_id", how="left")
    cand_final["matched_entity_ids"] = cand_final["matched_entity_ids"].fillna("")
    cand_final.to_csv(os.path.join(OUT_DIR, "matching_results.tsv"), sep="\t", index=False)
    
    cand_final.rename(columns={"matched_entity_ids": "candidate_entity_ids"}).to_csv(os.path.join(OUT_DIR, "candidate_pairs.tsv"), sep="\t", index=False)
    
    print("DONE! Optimized output generated.")

if __name__ == "__main__":
    main()
