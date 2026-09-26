import pandas as pd
import os
import sys
import gc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "src"))
import preprocessing
import blocking
import features
import matching

def generate_candidates_chunked(s1_keys, s2_path, s3_path, chunksize=100000):
    all_cands = []
    
    for chunk in pd.read_csv(s2_path, sep="\t", dtype=str, usecols=["entity_id", "business_name", "business_address", "country"], chunksize=chunksize):
        chunk_keys = blocking.prep_keys(chunk)
        del chunk
        gc.collect()
        
        cands = blocking.generate_candidates(s1_keys, chunk_keys)
        all_cands.append(cands)
        del chunk_keys, cands
        gc.collect()
        
    for chunk in pd.read_csv(s3_path, sep="\t", dtype=str, usecols=["entity_id", "business_name", "business_address", "country"], chunksize=chunksize):
        chunk_keys = blocking.prep_keys(chunk)
        del chunk
        gc.collect()
        
        cands = blocking.generate_candidates(s1_keys, chunk_keys)
        all_cands.append(cands)
        del chunk_keys, cands
        gc.collect()
        
    if not all_cands:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
        
    final_cands = pd.concat(all_cands, ignore_index=True).drop_duplicates()
    return final_cands

def get_features_chunked(candidates_df, s1_path, s2_path, s3_path, chunksize=100000):
    needed_s1_ids = set(candidates_df["source1_entity_id"].unique())
    needed_s23_ids = set(candidates_df["candidate_entity_id"].unique())
    
    cols = ["entity_id", "business_name", "business_address", "country"]
    
    # Load S1 subset
    s1_accum = []
    for chunk in pd.read_csv(s1_path, sep="\t", dtype=str, usecols=cols, chunksize=chunksize):
        chunk = chunk[chunk["entity_id"].isin(needed_s1_ids)]
        if len(chunk) > 0:
            s1_accum.append(preprocessing.process_dataframe(chunk))
    s1_filtered = pd.concat(s1_accum, ignore_index=True) if s1_accum else pd.DataFrame()
    del s1_accum
    gc.collect()
    
    # Load S23 subset
    s23_accum = []
    for chunk in pd.read_csv(s2_path, sep="\t", dtype=str, usecols=cols, chunksize=chunksize):
        chunk = chunk[chunk["entity_id"].isin(needed_s23_ids)]
        if len(chunk) > 0:
            s23_accum.append(preprocessing.process_dataframe(chunk))
            
    for chunk in pd.read_csv(s3_path, sep="\t", dtype=str, usecols=cols, chunksize=chunksize):
        chunk = chunk[chunk["entity_id"].isin(needed_s23_ids)]
        if len(chunk) > 0:
            s23_accum.append(preprocessing.process_dataframe(chunk))
            
    s23_filtered = pd.concat(s23_accum, ignore_index=True) if s23_accum else pd.DataFrame()
    del s23_accum
    gc.collect()
    
    return features.calculate_features(candidates_df, s1_filtered, s23_filtered)

def main():
    print("--- 1. LOADING & BLOCKING TEST DATA ---")
    DATA_DIR = os.path.join(BASE_DIR, "../../dataset")
    OUT_DIR = os.path.join(BASE_DIR, "../../output")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    s1_test_path = os.path.join(DATA_DIR, "test/test_source1.tsv")
    s2_test_path = os.path.join(DATA_DIR, "test/test_source2.tsv")
    s3_test_path = os.path.join(DATA_DIR, "test/test_source3.tsv")
    
    # Extract S1 keys efficiently without keeping S1 in RAM
    s1_test_keys_accum = []
    for chunk in pd.read_csv(s1_test_path, sep="\t", dtype=str, usecols=["entity_id", "business_name", "business_address", "country"], chunksize=100000):
        s1_test_keys_accum.append(blocking.prep_keys(chunk))
    s1_test_keys = pd.concat(s1_test_keys_accum, ignore_index=True)
    del s1_test_keys_accum
    gc.collect()
    
    test_candidates = generate_candidates_chunked(s1_test_keys, s2_test_path, s3_test_path, chunksize=100000)
    print(f"Test Candidates Generated: {len(test_candidates)}")
    del s1_test_keys
    gc.collect()
    
    cand_out = test_candidates.groupby("source1_entity_id")["candidate_entity_id"].apply(lambda x: ",".join(x)).reset_index()
    cand_out.columns = ["source1_entity_id", "candidate_entity_ids"]
    
    # We need the original entity IDs to ensure no row is dropped
    s1_test_ids = pd.read_csv(s1_test_path, sep="\t", dtype=str, usecols=["entity_id"]).rename(columns={"entity_id": "source1_entity_id"})
    cand_out = s1_test_ids.merge(cand_out, on="source1_entity_id", how="left")
    cand_out["candidate_entity_ids"] = cand_out["candidate_entity_ids"].fillna("")
    cand_out.to_csv(os.path.join(OUT_DIR, "candidate_pairs.tsv"), sep="\t", index=False)
    del cand_out
    gc.collect()
    
    print("\n--- 2. LOADING & BLOCKING TRAIN DATA ---")
    s1_train_path = os.path.join(DATA_DIR, "train/train_source1.tsv")
    s2_train_path = os.path.join(DATA_DIR, "train/train_source2.tsv")
    s3_train_path = os.path.join(DATA_DIR, "train/train_source3.tsv")
    
    s1_train_keys_accum = []
    for chunk in pd.read_csv(s1_train_path, sep="\t", dtype=str, usecols=["entity_id", "business_name", "business_address", "country"], chunksize=100000):
        s1_train_keys_accum.append(blocking.prep_keys(chunk))
    s1_train_keys = pd.concat(s1_train_keys_accum, ignore_index=True)
    del s1_train_keys_accum
    gc.collect()
    
    train_candidates = generate_candidates_chunked(s1_train_keys, s2_train_path, s3_train_path, chunksize=100000)
    print(f"Train Candidates Generated: {len(train_candidates)}")
    del s1_train_keys
    gc.collect()
    
    print("\n--- 3. FEATURE ENGINEERING ---")
    train_features = get_features_chunked(train_candidates, s1_train_path, s2_train_path, s3_train_path)
    del train_candidates
    gc.collect()
    
    test_features = get_features_chunked(test_candidates, s1_test_path, s2_test_path, s3_test_path)
    del test_candidates
    gc.collect()
    
    print("\n--- 4. MODEL TRAINING ---")
    gt_train = pd.read_csv(os.path.join(DATA_DIR, "train/train_ground_truth.tsv"), sep="\t", dtype=str)
    model = matching.train_model(train_features, gt_train)
    del train_features, gt_train
    gc.collect()
    
    print("\n--- 5. MATCHING & RANKING (INFERENCE) ---")
    test_results = matching.predict_matches(model, test_features, threshold=0.75)
    del test_features
    gc.collect()
    
    final_out = s1_test_ids.merge(test_results, on="source1_entity_id", how="left")
    final_out["matched_entity_ids"] = final_out["matched_entity_ids"].fillna("")
    final_out.to_csv(os.path.join(OUT_DIR, "matching_results.tsv"), sep="\t", index=False)
    
    print("\n--- DONE ---")
    print(f"Results successfully saved to '{os.path.abspath(OUT_DIR)}'")

if __name__ == "__main__":
    main()
