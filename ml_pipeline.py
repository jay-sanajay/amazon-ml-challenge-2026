import pandas as pd
import numpy as np
import os
import gc
import re
from difflib import SequenceMatcher
from sklearn.ensemble import HistGradientBoostingClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../dataset")
OUT_DIR = os.path.join(BASE_DIR, "../../output")
os.makedirs(OUT_DIR, exist_ok=True)

# 1. MEMORY-SAFE CLEANING
def clean_data(df):
    df["name_clean"] = df["business_name"].fillna("").astype(str).str.lower().str.replace(r"[^\w\s]", "", regex=True)
    df["addr_clean"] = df["business_address"].fillna("").astype(str).str.lower().str.replace(r"[^\w\s]", "", regex=True)
    df["name_w1"] = df["name_clean"].str.split().str[0].fillna("")
    df["name_w2"] = df["name_clean"].str.split().str[1].fillna("")
    df["zipcode"] = df["business_address"].fillna("").astype(str).str.extract(r"(\b\d{5,6}\b)")
    df["zipcode"] = df["zipcode"].fillna("")
    df["country"] = df["country"].fillna("UNKNOWN")
    df["addr_w1"] = df["addr_clean"].str.split().str[0].fillna("")
    return df[["entity_id", "country", "name_w1", "name_w2", "zipcode", "name_clean", "addr_clean", "addr_w1"]]

# 2. HIGH-RECALL BLOCKING
def block_chunk(s1_df, c_df):
    def block(keys, max_freq=50):
        b1 = s1_df[s1_df[keys].notna().all(axis=1) & (s1_df[keys] != "").all(axis=1)]
        b2 = c_df[c_df[keys].notna().all(axis=1) & (c_df[keys] != "").all(axis=1)]
        if b1.empty or b2.empty: return pd.DataFrame()
        
        k1 = b1.groupby(keys).size().reset_index(name='c')
        v1 = k1[k1['c'] <= max_freq].drop(columns=['c'])
        k2 = b2.groupby(keys).size().reset_index(name='c')
        v2 = k2[k2['c'] <= max_freq].drop(columns=['c'])
        vk = pd.merge(v1, v2, on=keys, how="inner")
        
        if not vk.empty:
            m1 = pd.merge(b1, vk, on=keys, how="inner")
            m2 = pd.merge(b2, vk, on=keys, how="inner")
            cols1 = list(set(["entity_id", "name_clean", "addr_clean"] + keys))
            cols2 = list(set(["entity_id", "name_clean", "addr_clean"] + keys))
            return pd.merge(m1[cols1], m2[cols2], on=keys)
        return pd.DataFrame()

    p1 = block(["country", "name_w1", "name_w2"], 500)
    p2 = block(["country", "name_w1", "zipcode"], 500)
    p3 = block(["country", "name_clean"], 2000)
    p4 = block(["country", "addr_w1", "zipcode"], 500)
    
    cands = [p for p in [p1, p2, p3, p4] if not p.empty]
    if cands:
        return pd.concat(cands, ignore_index=True).drop_duplicates(subset=["entity_id_x", "entity_id_y"])
    return pd.DataFrame()

# 3. ADVANCED ML FEATURE EXTRACTION
def extract_numbers(s):
    if not s: return set()
    return set(re.findall(r"\d+", str(s)))

def calculate_features(cands):
    from rapidfuzz import fuzz
    
    def fast_sim(l1, l2):
        return [fuzz.ratio(str(a), str(b)) / 100.0 for a, b in zip(l1, l2)]
    
    cands["name_sim"] = fast_sim(cands["name_clean_x"], cands["name_clean_y"])
    cands["addr_sim"] = fast_sim(cands["addr_clean_x"], cands["addr_clean_y"])
    
    # Exact number matches (crucial for building/street logic)
    n1 = [extract_numbers(a) for a in cands["addr_clean_x"]]
    n2 = [extract_numbers(a) for a in cands["addr_clean_y"]]
    
    cands["num_overlap"] = [
        len(a & b) / len(a | b) if (a or b) else 1.0 
        for a, b in zip(n1, n2)
    ]
    return cands[["entity_id_x", "entity_id_y", "name_sim", "addr_sim", "num_overlap"]]

# 4. CHUNK PROCESSING
def build_dataset(s1_path, s2_path, s3_path, is_train=False, gt_df=None):
    s1_accum = []
    for chunk in pd.read_csv(s1_path, sep="\t", dtype=str, chunksize=100000, quoting=3):
        s1_accum.append(clean_data(chunk))
    s1 = pd.concat(s1_accum, ignore_index=True)
    del s1_accum; gc.collect()
    
    features_list = []
    for path in [s2_path, s3_path]:
        for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=100000, quoting=3):
            c_df = clean_data(chunk)
            cands = block_chunk(s1, c_df)
            del c_df; gc.collect()
            
            if not cands.empty:
                feats = calculate_features(cands)
                features_list.append(feats)
            del cands; gc.collect()
            
    final_features = pd.concat(features_list, ignore_index=True).drop_duplicates(subset=["entity_id_x", "entity_id_y"])
    
    if is_train and gt_df is not None:
        # Create ground truth labels
        gt_pairs = set()
        for _, row in gt_df.dropna(subset=['matched_entity_ids']).iterrows():
            s1_id = row['source1_entity_id']
            for s2_id in str(row['matched_entity_ids']).split(','):
                gt_pairs.add((s1_id, s2_id.strip()))
                
        final_features['label'] = final_features.apply(
            lambda x: 1 if (x['entity_id_x'], x['entity_id_y']) in gt_pairs else 0, 
            axis=1
        )
    return final_features

# 5. HYPERTUNING & F0.5 SCORING
def calculate_f05(gt_map, pred_map):
    scores = []
    for s1, true_matches in gt_map.items():
        pred_matches = pred_map.get(s1, set())
        if not true_matches:
            scores.append(1.0 if not pred_matches else 0.0)
            continue
        tp = len(true_matches & pred_matches)
        fp = len(pred_matches - true_matches)
        fn = len(true_matches - pred_matches)
        if tp == 0:
            scores.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        scores.append((1.25 * precision * recall) / (0.25 * precision + recall))
    return sum(scores) / len(scores) if scores else 0

def tune_threshold(model, val_features, gt_df):
    print("Hypertuning Threshold to Maximize F0.5...")
    X_val = val_features[["name_sim", "addr_sim", "num_overlap"]]
    probs = model.predict_proba(X_val)[:, 1]
    
    # Build validation ground truth map
    gt_map = {}
    val_s1 = set(val_features["entity_id_x"].unique())
    for _, row in gt_df[gt_df["source1_entity_id"].isin(val_s1)].iterrows():
        matches = str(row["matched_entity_ids"]).split(",") if pd.notna(row["matched_entity_ids"]) else []
        gt_map[row["source1_entity_id"]] = set([m.strip() for m in matches if m.strip()])

    best_thresh, best_f05 = 0.50, 0.0
    for thresh in np.arange(0.50, 0.99, 0.02):
        preds = (probs >= thresh).astype(int)
        
        pred_map = {}
        for i, is_match in enumerate(preds):
            if is_match:
                s1_id = val_features.iloc[i]["entity_id_x"]
                s2_id = val_features.iloc[i]["entity_id_y"]
                if s1_id not in pred_map: pred_map[s1_id] = set()
                pred_map[s1_id].add(s2_id)
                
        score = calculate_f05(gt_map, pred_map)
        print(f"Threshold: {thresh:.2f} | F0.5 Score: {score:.5f}")
        if score > best_f05:
            best_f05 = score
            best_thresh = thresh
            
    print(f"Optimal Threshold Found: {best_thresh:.2f} (F0.5 = {best_f05:.5f})")
    return best_thresh

def main():
    print("--- 1. BUILDING ML TRAINING DATA ---")
    train_gt = pd.read_csv(os.path.join(DATA_DIR, "train/train_ground_truth.tsv"), sep="\t", dtype=str)
    
    # We use the first 30% of Train data to train the model to save time and RAM
    train_features = build_dataset(
        os.path.join(DATA_DIR, "train/train_source1.tsv"),
        os.path.join(DATA_DIR, "train/train_source2.tsv"),
        os.path.join(DATA_DIR, "train/train_source3.tsv"),
        is_train=True, gt_df=train_gt
    )
    
    print("--- 2. TRAINING AI MODEL ---")
    X_train = train_features[["name_sim", "addr_sim", "num_overlap"]]
    y_train = train_features["label"]
    model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    print("--- 3. HYPERTUNING ---")
    # We tune on the training set features we just generated
    optimal_thresh = tune_threshold(model, train_features, train_gt)
    del train_features, X_train, y_train; gc.collect()
    
    print("--- 4. PROCESSING TEST SET ---")
    test_features = build_dataset(
        os.path.join(DATA_DIR, "test/test_source1.tsv"),
        os.path.join(DATA_DIR, "test/test_source2.tsv"),
        os.path.join(DATA_DIR, "test/test_source3.tsv"),
        is_train=False
    )
    
    print(f"--- 5. INFERENCE (Applying Optimal Threshold: {optimal_thresh:.2f}) ---")
    X_test = test_features[["name_sim", "addr_sim", "num_overlap"]]
    test_probs = model.predict_proba(X_test)[:, 1]
    
    matches = test_features[test_probs >= optimal_thresh]
    cand_out = matches.groupby("entity_id_x")["entity_id_y"].apply(lambda x: ",".join(x)).reset_index()
    cand_out.columns = ["source1_entity_id", "matched_entity_ids"]
    
    s1_ids = pd.read_csv(os.path.join(DATA_DIR, "test/test_source1.tsv"), sep="\t", dtype=str, usecols=["entity_id"]).rename(columns={"entity_id": "source1_entity_id"})
    final_out = s1_ids.merge(cand_out, on="source1_entity_id", how="left")
    final_out["matched_entity_ids"] = final_out["matched_entity_ids"].fillna("")
    
    final_out.to_csv(os.path.join(OUT_DIR, "matching_results.tsv"), sep="\t", index=False)
    final_out.rename(columns={"matched_entity_ids": "candidate_entity_ids"}).to_csv(os.path.join(OUT_DIR, "candidate_pairs.tsv"), sep="\t", index=False)
    print("DONE! ML pipeline completed successfully.")

if __name__ == "__main__":
    main()
