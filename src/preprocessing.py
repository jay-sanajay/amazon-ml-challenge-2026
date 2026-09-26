import pandas as pd

def process_dataframe(df):
    """
    Fully vectorized preprocessing to prevent memory fragmentation and ArrayMemoryError.
    Avoids .apply() which creates millions of Python objects.
    """
    df = df.copy()
    
    # 1. Clean Business Name
    name = df["business_name"].fillna("").astype(str).str.lower()
    # Remove accents using normalize (if supported, otherwise skip and rely on regex)
    # Fast regex replacement for punctuation and special chars
    name = name.str.replace(r"[^\w\s]", " ", regex=True)
    
    # Remove legal suffixes via vectorized regex
    suffixes = r"\b(ltd|limited|pvt|private|inc|llc|corp|corporation)\b"
    name = name.str.replace(suffixes, "", regex=True)
    
    # Remove extra spaces
    name = name.str.replace(r"\s+", " ", regex=True).str.strip()
    df["norm_name"] = name
    
    # 2. Clean Business Address
    addr = df["business_address"].fillna("").astype(str).str.lower()
    addr = addr.str.replace(r"[^\w\s]", " ", regex=True)
    addr = addr.str.replace(r"\s+", " ", regex=True).str.strip()
    df["norm_address"] = addr
    
    # 3. Country
    df["country"] = df["country"].fillna("UNKNOWN").str.upper()
    
    return df
