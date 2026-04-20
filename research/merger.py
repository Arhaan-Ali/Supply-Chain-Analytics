import pandas as pd

train = pd.read_csv("../Data/train.csv")
test = pd.read_csv("../Data/Processed Data/test_filled.csv")
store = pd.read_csv("../Data/Processed Data/store_filled.csv")

train_merged = train.merge(store, on='Store', how='left')
test_merged = test.merge(store, on='Store', how='left')

train_merged.to_csv("../Data/Processed Data/train_merged.csv", index=False)
test_merged.to_csv("../Data/Processed Data/test_merged.csv", index=False)