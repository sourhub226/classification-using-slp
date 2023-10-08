import pandas as pd
import matplotlib.pyplot as plt

dataset = "rice.csv"
df = pd.read_csv(dataset)
# Assuming your DataFrame is called 'df'
# Identify the last column (class labels)
last_column = df.columns[-1]

# Exclude the last column from your data
data = df.iloc[:, :-1]

# Get the unique class labels
unique_classes = df[last_column].unique()

# Generate scatter plots for every pair of features, grouping by class
for i in range(len(data.columns)):
    for j in range(i + 1, len(data.columns)):
        plt.figure(figsize=(8, 6))
        for cls in unique_classes:
            class_data = data[df[last_column] == cls]
            plt.scatter(
                class_data.iloc[:, i], class_data.iloc[:, j], label=f"Class {cls}"
            )
        plt.xlabel(data.columns[i])
        plt.ylabel(data.columns[j])
        plt.title(f"Scatter plot between {data.columns[i]} and {data.columns[j]}")
        plt.legend()
        plt.show()
