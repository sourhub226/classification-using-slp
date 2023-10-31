import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Loading the dataset
def load_dataset(filename):
    data = np.genfromtxt(filename, delimiter=",")
    data = np.delete(data, 0, axis=0)
    # print(data)
    X = data[:, :-1]  # feature columns
    y = data[:, -1]  # class column
    return X, y


# Preprocessing the data (normalization and adding bias col)
def preprocess_data(X):
    X[np.isnan(X)] = 0
    # min-max normalization
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    X_normalised = (X - min_vals) / (max_vals - min_vals)
    # print(X_normalised)

    # or use this z-score normalization
    # X_normalised = (X - X.mean(axis=0)) / X.std(axis=0)
    bias_col = np.ones((X.shape[0], 1))
    return np.concatenate((bias_col, X_normalised), axis=1)


# Prediction using threshold function
def predict(X, weights):
    return np.where(np.dot(X, weights) > 0, 1, 0)


# Single layer perceptron model
def slp_train(X, y, learning_rate, epochs):
    weights = np.zeros(X.shape[1])  # Initialize weights to zeros
    losses = []  # To store loss at each epoch

    for _ in range(epochs):
        total_loss = 0
        for i in range(X.shape[0]):  # Loop through each data point
            prediction = predict(X[i], weights)
            error = y[i] - prediction
            weights += learning_rate * error * X[i]
            total_loss += error**2
        losses.append(total_loss)

    return weights, losses


# Calculating accuracy
def calc_accuracy(X, y, weights):
    y_pred = predict(X, weights)
    return np.mean(y == y_pred)


# -----start here-----
# parameters
learning_rate = 1
epochs = 200

X, y = load_dataset("datasets/ionosphere_pre.csv")
X = preprocess_data(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Training SLP on the training data
weights, losses = slp_train(
    X_train, y_train, learning_rate=learning_rate, epochs=epochs
)
print(f"W* = {weights}")

# Calculating accuracy on training data and testing data
print(
    f"Accuracy on training data = {calc_accuracy(X_train,y_train,weights) * 100:.2f}%"
)
print(f"Accuracy on testing data = {calc_accuracy(X_test,y_test,weights) * 100:.2f}%")


# ploting loss vs epochs
plt.plot(range(len(losses)), losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs. Epochs")
plt.show()
