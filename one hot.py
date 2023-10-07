import pandas as pd

dataset = "datasets/bmi.csv"
df = pd.read_csv(dataset)

# spliting df into X and y
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# processing X
cat_cols = X.select_dtypes(exclude=["number"]).columns
X_processed = pd.get_dummies(X, columns=cat_cols, dtype=int)

# processing y
unique_classes = y.unique()
class_mapping = {class_name: index for index, class_name in enumerate(unique_classes)}
print(class_mapping)
y_processed = y.replace(class_mapping)

# concatenating X and y into df
df_processed = pd.concat([X_processed, y_processed], axis=1)
print(df_processed)

# saving processed dataset
df_processed.to_csv(f'{dataset.replace(".csv","")}_preprocessed.csv', index=False)
