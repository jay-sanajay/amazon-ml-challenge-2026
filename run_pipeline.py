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

def generate_candidates_chunked(s1_df, s2_path, s3_path, chunksize=100000):
    all_cands = []
    
    # Process Source 2 in chunks
    for chunk in pd.read_csv(s2_path, sep="\t", dtype=str, chunksize=chunksize):
        chunk = preprocessing.process_dataframe(chunk)
        cands = blocking.generate_candidates(s1_df, chunk)
        all_cands.append(cands)
        del chunk, cands
        gc.collect()
        
    # Process Source 3 in chunks
    for chunk in pd.read_csv(s3_path, sep="\t", dtype=str, chunksize=chunksize):
        chunk = preprocessing.process_dataframe(chunk)
        cands = blocking.generate_candidates(s1_df, chunk)
        all_cands.append(cands)
        del chunk, cands
        gc.collect()
        
    if not all_cands:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
        
    final_cands = pd.concat(all_cands, ignore_index=True).drop_duplicates()
    return final_cands

def get_features_chunked(candidates_df, s1_df, s2_path, s3_path, chunksize=100000):
    # To get features, we need the raw text. We can read S2/S3 in chunks, 
    # filter only the ones that are in candidates, and accumulate.
    needed_s23_ids = set(candidates_df["candidate_entity_id"].unique())
    s23_accum = []
    
    cols = ["entity_id", "business_name", "business_address", "country"]
    
    for chunk in pd.read_csv(s2_path, sep="\t", dtype=str, usecols=cols, chunksize=chunksize):
        chunk = chunk[chunk["entity_id"].isin(needed_s23_ids)]
        if len(chunk) > 0:
            s23_accum.append(preprocessing.process_dataframe(chunk))
            
    for chunk in pd.read_csv(s3_path, sep="\t", dtype=str, usecols=cols, chunksize=chunksize):
        chunk = chunk[chunk["entity_id"].isin(needed_s23_ids)]
        if len(chunk) > 0:
            s23_accum.append(preprocessing.process_dataframe(chunk))
            
    if not s23_accum:
        return pd.DataFrame()
        
    s23_filtered = pd.concat(s23_accum, ignore_index=True)
    return features.calculate_features(candidates_df, s1_df, s23_filtered)

def main():
    print("--- 1. LOADING & BLOCKING TEST DATA ---")
    DATA_DIR = os.path.join(BASE_DIR, "../../dataset")
    OUT_DIR = os.path.join(BASE_DIR, "../../output")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    s1_test = pd.read_csv(os.path.join(DATA_DIR, "test/test_source1.tsv"), sep="\t", dtype=str)
    s1_test = preprocessing.process_dataframe(s1_test)
    
    test_candidates = generate_candidates_chunked(
        s1_test,
        os.path.join(DATA_DIR, "test/test_source2.tsv"),
        os.path.join(DATA_DIR, "test/test_source3.tsv"),
        chunksize=250000
    )
    print(f"Test Candidates Generated: {len(test_candidates)}")
    
    cand_out = test_candidates.groupby("source1_entity_id")["candidate_entity_id"].apply(lambda x: ",".join(x)).reset_index()
    cand_out.columns = ["source1_entity_id", "candidate_entity_ids"]
    cand_out = s1_test[["entity_id"]].rename(columns={"entity_id": "source1_entity_id"}).merge(cand_out, on="source1_entity_id", how="left")
    cand_out["candidate_entity_ids"] = cand_out["candidate_entity_ids"].fillna("")
    cand_out.to_csv(os.path.join(OUT_DIR, "candidate_pairs.tsv"), sep="\t", index=False)
    
    print("\n--- 2. LOADING & BLOCKING TRAIN DATA ---")
    s1_train = pd.read_csv(os.path.join(DATA_DIR, "train/train_source1.tsv"), sep="\t", dtype=str)
    s1_train = preprocessing.process_dataframe(s1_train)
    gt_train = pd.read_csv(os.path.join(DATA_DIR, "train/train_ground_truth.tsv"), sep="\t", dtype=str)
    
    train_candidates = generate_candidates_chunked(
        s1_train,
        os.path.join(DATA_DIR, "train/train_source2.tsv"),
        os.path.join(DATA_DIR, "train/train_source3.tsv"),
        chunksize=250000
    )
    print(f"Train Candidates Generated: {len(train_candidates)}")
    
    print("\n--- 3. FEATURE ENGINEERING ---")
    train_features = get_features_chunked(
        train_candidates, s1_train,
        os.path.join(DATA_DIR, "train/train_source2.tsv"),
        os.path.join(DATA_DIR, "train/train_source3.tsv")
    )
    del train_candidates, s1_train
    gc.collect()
    
    test_features = get_features_chunked(
        test_candidates, s1_test,
        os.path.join(DATA_DIR, "test/test_source2.tsv"),
        os.path.join(DATA_DIR, "test/test_source3.tsv")
    )
    del test_candidates
    gc.collect()
    
    print("\n--- 4. MODEL TRAINING ---")
    model = matching.train_model(train_features, gt_train)
    del train_features, gt_train
    gc.collect()
    
    print("\n--- 5. MATCHING & RANKING (INFERENCE) ---")
    test_results = matching.predict_matches(model, test_features, threshold=0.75)
    
    final_out = s1_test[["entity_id"]].rename(columns={"entity_id": "source1_entity_id"}).merge(test_results, on="source1_entity_id", how="left")
    final_out["matched_entity_ids"] = final_out["matched_entity_ids"].fillna("")
    final_out.to_csv(os.path.join(OUT_DIR, "matching_results.tsv"), sep="\t", index=False)
    
    print("\n--- DONE ---")
    print(f"Results successfully saved to '{os.path.abspath(OUT_DIR)}'")

if __name__ == "__main__":
    main()
