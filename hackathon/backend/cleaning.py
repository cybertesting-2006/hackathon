
import pandas as pd, numpy as np

def clean_data(df):
    df = df.copy().drop_duplicates().dropna(axis=1, how="all")
    df = df.loc[:, df.nunique() > 1]  # drop constant cols
    # fix strings
    obj = df.select_dtypes(include="object").columns
    df[obj] = df[obj].apply(lambda c: c.astype(str).str.strip().str.lower())
    # fill missing
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].apply(lambda c: c.fillna(c.mean()))
    cat = df.select_dtypes(exclude=[np.number]).columns
    df[cat] = df[cat].apply(lambda c: c.fillna(c.mode().iloc[0] if not c.mode().empty else "unknown"))
    # type inference
    for col in df.columns:
        for func in (pd.to_numeric, pd.to_datetime):
            try: df[col] = func(df[col]); break
            except: pass
    return df

def encode_data(df): return pd.get_dummies(df, drop_first=True)

def detect_outliers(df, z=3):
    return {c: df.index[((df[c]-df[c].mean())/df[c].std()).abs() > z].tolist()
            for c in df.select_dtypes(include=[np.number])}

def remove_outliers(df, z=3):
    mask = ((df.select_dtypes(include=[np.number]) - df.mean())/df.std()).abs().le(z).all(1)
    return df[mask]
