import pandas as pd
from imblearn.over_sampling import SMOTE
import re
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/spam.csv", encoding="latin-1")
print(df.head())
print(df["v2"])
print(df.head())
print(df.columns)
df = df.rename(columns={"v1": "label", "v2": "text"})
df["label"] = df["label"].map({"ham": 0, "spam": 1})
df = df[["label", "text"]]
print(df["label"].value_counts())
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # keep only letters
    text = text.strip()
    return text

df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(X_train.head())
