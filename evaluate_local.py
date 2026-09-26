import pandas as pd
import os
import gc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../dataset")

def calculate_f05(gt_map, pred_map):
    f05_scores = []
    for s1, true_matches in gt_map.items():
        pred_matches = pred_map.get(s1, set())
        if len(true_matches) == 0:
            f05_scores.append(1.0 if len(pred_matches) == 0 else 0.0)
            continue
        tp = len(true_matches.intersection(pred_matches))
        fp = len(pred_matches - true_matches)
        fn = len(true_matches - pred_matches)
        if tp == 0:
            f05_scores.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
        f05_scores.append(f05)
    return sum(f05_scores) / len(f05_scores) if f05_scores else 0

def clean_data(df):
    df["name_clean"] = df["business_name"].fillna("").astype(str).str.lower().str.replace(r"[^\w\s]", "", regex=True)
    df["addr_clean"] = df["business_address"].fillna("").astype(str).str.lower().str.replace(r"[^\w\s]", "", regex=True)
    
    name_splits = df["name_clean"].str.split()
    df["name_w1"] = name_splits.str[0].fillna("")
    df["name_w2"] = name_splits.str[1].fillna("")
    
    df["zipcode"] = df["business_address"].fillna("").astype(str).str.extract(r"(\b\d{5,6}\b)")
    df["zipcode"] = df["zipcode"].fillna("")
    
    df["country"] = df["country"].fillna("UNKNOWN")
    return df[["entity_id", "country", "name_w1", "name_w2", "zipcode", "name_clean", "addr_clean"]]

def main():
    print("--- LOCAL EVALUATION ON TRAIN DATA ---")
    s1_path = os.path.join(DATA_DIR, "train/train_source1.tsv")
    s2_path = os.path.join(DATA_DIR, "train/train_source2.tsv")
    
    print("Loading Subset of Train Data...")
    s1 = clean_data(pd.read_csv(s1_path, sep="\t", dtype=str, usecols=["entity_id", "business_name", "business_address", "country"], nrows=50000))
    s2 = clean_data(pd.read_csv(s2_path, sep="\t", dtype=str, usecols=["entity_id", "business_name", "business_address", "country"], nrows=100000))
    gt = pd.read_csv(os.path.join(DATA_DIR, "train/train_ground_truth.tsv"), sep="\t", dtype=str)
    
    print("Applying Rules...")
    def block(keys):
        b1 = s1[s1[keys].notna().all(axis=1) & (s1[keys] != "").all(axis=1)]
        b2 = s2[s2[keys].notna().all(axis=1) & (s2[keys] != "").all(axis=1)]
        if b1.empty or b2.empty: return pd.DataFrame()
        
        k1 = b1.groupby(keys).size().reset_index(name='c')
        v1 = k1[k1['c'] <= 50].drop(columns=['c'])
        k2 = b2.groupby(keys).size().reset_index(name='c')
        v2 = k2[k2['c'] <= 50].drop(columns=['c'])
        vk = pd.merge(v1, v2, on=keys, how="inner")
        
        if len(vk) > 0:
            m1 = pd.merge(b1, vk, on=keys, how="inner")
            m2 = pd.merge(b2, vk, on=keys, how="inner")
            cols1 = list(set(["entity_id", "country", "name_clean", "addr_clean"] + keys))
            cols2 = list(set(["entity_id", "country", "name_clean", "addr_clean"] + keys))
            return pd.merge(m1[cols1], m2[cols2], on=keys)
        return pd.DataFrame()
        
    p1 = block(["country", "name_w1", "name_w2"])
    p2 = block(["country", "name_w1", "zipcode"])
    
    cands_list = [p for p in [p1, p2] if not p.empty]
    if cands_list:
        cands = pd.concat(cands_list, ignore_index=True).drop_duplicates(subset=["entity_id_x", "entity_id_y"])
        n1_sets = cands["name_clean_x"].str.split().apply(set)
        n2_sets = cands["name_clean_y"].str.split().apply(set)
        a1_sets = cands["addr_clean_x"].str.split().apply(set)
        a2_sets = cands["addr_clean_y"].str.split().apply(set)
        
        n_inter = [len(a & b) for a, b in zip(n1_sets, n2_sets)]
        n_union = [len(a | b) for a, b in zip(n1_sets, n2_sets)]
        cands["name_score"] = [i / u if u > 0 else 0 for i, u in zip(n_inter, n_union)]
        
        a_inter = [len(a & b) for a, b in zip(a1_sets, a2_sets)]
        a_union = [len(a | b) for a, b in zip(a1_sets, a2_sets)]
        cands["addr_score"] = [i / u if u > 0 else 0 for i, u in zip(a_inter, a_union)]
        
        strict_match = cands[
            (cands["name_score"] >= 0.80) | 
            ((cands["name_score"] >= 0.50) & (cands["addr_score"] >= 0.25))
        ]
        matches = strict_match[["entity_id_x", "entity_id_y"]]
    else:
        matches = pd.DataFrame(columns=["entity_id_x", "entity_id_y"])
        
    print("Scoring...")
    pred_map = {}
    for _, row in matches.iterrows():
        s1_id, s2_id = row["entity_id_x"], row["entity_id_y"]
        if s1_id not in pred_map: pred_map[s1_id] = set()
        pred_map[s1_id].add(s2_id)
        
    gt_map = {}
    val_s1 = set(s1["entity_id"].unique())
    val_s2 = set(s2["entity_id"].unique())
    val_gt = gt[gt["source1_entity_id"].isin(val_s1)]
    
    for _, row in val_gt.iterrows():
        true_matches = str(row["matched_entity_ids"]).split(",") if pd.notna(row["matched_entity_ids"]) else []
        valid_matches = set([m.strip() for m in true_matches if m.strip() and m.strip() in val_s2])
        gt_map[row["source1_entity_id"]] = valid_matches
        
    score = calculate_f05(gt_map, pred_map)
    print(f"\n======================================")
    print(f"ESTIMATED F0.5 SCORE: {score:.4f}")
    print(f"======================================")

if __name__ == "__main__":
    main()
