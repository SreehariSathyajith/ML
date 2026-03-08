import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)

    ent = 0
    for p in probs:
        ent -= p * np.log2(p)

    return ent

def equal_width_binning(series, bins=4):
    series = pd.to_numeric(series, errors='coerce')
    series = series.dropna()
    
    if len(series) == 0:
        return pd.Series([], dtype='int64')
    
    min_val = series.min()
    max_val = series.max()
    if min_val == max_val:
        return pd.Series([0] * len(series), index=series.index)
    
    width = (max_val - min_val) / bins
    bins_edges = [min_val + i * width for i in range(bins + 1)]
    
    return pd.cut(series, bins=bins_edges, labels=False, include_lowest=True)

def gini_index(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    gini = 1 - np.sum(probs ** 2)
    return gini

def information_gain(X_col, y):
    parent_entropy = entropy(y)
    values, counts = np.unique(X_col, return_counts=True)
    weighted_entropy = 0
    mask = ~pd.isna(X_col)
    X_col_clean = X_col[mask]
    y_clean = y[mask]
    
    values, counts = np.unique(X_col_clean, return_counts=True)
    
    for v, c in zip(values, counts):
        subset_y = y_clean[X_col_clean == v]
        if len(subset_y) > 0:
            weighted_entropy += (c / len(X_col_clean)) * entropy(subset_y)
    
    return parent_entropy - weighted_entropy

def find_root_feature(X, y):
    gains = {}
    
    for col in X.columns:
        try:
            binned = equal_width_binning(X[col])
            if len(binned) > 0:
                gains[col] = information_gain(binned, y)
            else:
                gains[col] = 0
        except Exception as e:
            print(f"Error processing column {col}: {e}")
            gains[col] = 0
    
    if not gains:
        return None, {}
    
    root = max(gains, key=gains.get)
    return root, gains

def main():
    RAW_EEG_FOLDER = r"C:\Users\sreeh\ML_PROJ\EEG_Raw_CSV"
    PARTICIPANTS_FILE = r"C:\Users\sreeh\ML_PROJ\participants.tsv"
    participants = pd.read_csv(PARTICIPANTS_FILE, sep="\t")
    features = []
    
    for file in sorted(os.listdir(RAW_EEG_FOLDER)):
        if not file.endswith(".csv"):
            continue

        subject_id = file.replace("_rawEEG.csv", "")
        path = os.path.join(RAW_EEG_FOLDER, file)
        df = pd.read_csv(path, nrows=60000)

        row = {"participant_id": subject_id}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            row[col] = df[col].mean(skipna=True)

        features.append(row)
    
    feature_df = pd.DataFrame(features)
    X = feature_df.drop('participant_id', axis=1)
    X = X.select_dtypes(include=[np.number])
    X = X.dropna(axis=1, how='all')
    
    X = X.fillna(X.mean())
    
    y = participants["MMSE"]
    
    root_feature, gains = find_root_feature(X, y)
    print("Root Feature:", root_feature)
    print("Information Gains:")
    for feature, gain in gains.items():
        print(f"{feature}: {gain:.4f}")
    
    print("Dataset Entropy:", entropy(y))
    print("Dataset Gini Index:", gini_index(y))
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X, y_encoded)
    
    plt.figure(figsize=(15,10))
    plot_tree(dt,
            feature_names=X.columns.tolist(),
            class_names=[str(cls) for cls in le.classes_],
            filled=True,
            rounded=True,
            fontsize=10)
    plt.title("Decision Tree (Max Depth = 4)")
    plt.tight_layout()
    plt.show()

    if X.shape[1] >= 2:
        features_2 = X.iloc[:, :2].values
        dt2 = DecisionTreeClassifier(max_depth=4, random_state=42)
        dt2.fit(features_2, y_encoded)
        
        x_min, x_max = features_2[:,0].min() - 1, features_2[:,0].max() + 1
        y_min, y_max = features_2[:,1].min() - 1, features_2[:,1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )
        
        Z = dt2.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        scatter = plt.scatter(features_2[:,0], features_2[:,1], c=y_encoded, 
                            cmap='RdYlBu', edgecolors='black', linewidth=1)
        plt.xlabel(X.columns[0])
        plt.ylabel(X.columns[1])
        plt.title("Decision Boundary (First Two Features)")
        plt.colorbar(scatter)
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough features for decision boundary plot ")

if __name__ == "__main__":
    main()