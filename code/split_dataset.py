import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/cleaned_dataset.csv")

train_df, test_df = train_test_split(df, test_size=0.25, random_state=42, shuffle=True)

train_df.to_csv("data/train_dataset.csv", index=False)
test_df.to_csv("data/test_dataset.csv", index=False)

print("Training set size:", len(train_df))
print("Testing set size:", len(test_df))
print("Files saved as train_dataset.csv and test_dataset.csv")