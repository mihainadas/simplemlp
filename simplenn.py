import random
import math
import json
import os
from tqdm import tqdm

# import texts and labels from data.jsonl
texts, labels = [], []
with open(os.path.dirname(__file__) + "/output/jsonl/data.jsonl") as f:
    for line in f:
        data = json.loads(line)
        texts.append(data["text"])
        labels.append(int(data["label"]))  # convert label from string to integer


# Simple text preprocessing: tokenize text and create a vocabulary
def tokenize(text):
    return text.lower().split()


# Create a vocabulary mapping each word to a unique index
vocab = {}
for text in texts:
    for word in tokenize(text):
        if word not in vocab:
            vocab[word] = len(vocab)


# Convert each text to a binary vector indicating the presence of words from the vocabulary
def text_to_vector(text):
    vector = [0] * len(vocab)
    words = tokenize(text)
    for word in words:
        if word in vocab:
            vector[vocab[word]] = 1
    return vector


# Define a function to convert a binary vector back to text
def vector_to_text(vector):
    text = ""
    for i, value in enumerate(vector):
        if value == 1:
            for word, index in vocab.items():
                if index == i:
                    text += word + " "
    return text


# Convert texts to vectors
X = []
for text in texts:
    X.append(text_to_vector(text))
y = labels


# Split the dataset into training and testing sets
def train_test_split(X, y, test_size=0.25):
    indices = list(range(len(X)))
    random.shuffle(indices)  # Shuffle the indices to randomize the split
    split_idx = int(len(X) * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for i in train_indices:
        X_train.append(X[i])
        y_train.append(y[i])

    for i in test_indices:
        X_test.append(X[i])
        y_test.append(y[i])

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# Define a simple feedforward neural network
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases with random values
        self.W1 = []
        for _ in range(input_size):
            self.W1.append([random.random() for _ in range(hidden_size)])
        self.b1 = [0] * hidden_size

        self.W2 = []
        for _ in range(hidden_size):
            self.W2.append([random.random() for _ in range(output_size)])
        self.b2 = [0] * output_size

    # Sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    # Derivative of the sigmoid function for backpropagation
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # Forward pass: compute activations of the hidden and output layers
    def forward(self, x):
        self.z1 = []
        for j in range(len(self.b1)):
            z1_value = sum(x[i] * self.W1[i][j] for i in range(len(x))) + self.b1[j]
            self.z1.append(z1_value)
        self.a1 = [self.sigmoid(z) for z in self.z1]

        self.z2 = []
        for k in range(len(self.b2)):
            z2_value = (
                sum(self.a1[j] * self.W2[j][k] for j in range(len(self.a1)))
                + self.b2[k]
            )
            self.z2.append(z2_value)
        self.a2 = [self.sigmoid(z) for z in self.z2]

        return self.a2

    # Backward pass: update weights and biases based on the error
    def backward(self, x, y, output, learning_rate):
        # Calculate error and delta for output layer
        output_error = []
        output_delta = []
        for i in range(len(y)):
            output_error.append(y[i] - output[i])
            output_delta.append(output_error[i] * self.sigmoid_derivative(output[i]))

        # Calculate error and delta for hidden layer
        hidden_error = []
        for j in range(len(self.a1)):
            error_sum = 0
            for k in range(len(output_delta)):
                error_sum += output_delta[k] * self.W2[j][k]
            hidden_error.append(error_sum)

        hidden_delta = []
        for j in range(len(hidden_error)):
            hidden_delta.append(hidden_error[j] * self.sigmoid_derivative(self.a1[j]))

        # Update weights and biases for the output layer
        for j in range(len(self.W2)):
            for k in range(len(self.W2[j])):
                self.W2[j][k] += self.a1[j] * output_delta[k] * learning_rate

        for k in range(len(self.b2)):
            self.b2[k] += output_delta[k] * learning_rate

        # Update weights and biases for the hidden layer
        for i in range(len(self.W1)):
            for j in range(len(self.W1[i])):
                self.W1[i][j] += x[i] * hidden_delta[j] * learning_rate

        for j in range(len(self.b1)):
            self.b1[j] += hidden_delta[j] * learning_rate

    # Train the neural network for a specified number of epochs
    def train(self, X, y, epochs, learning_rate):
        for epoch in tqdm(range(epochs)):
            for xi, yi in zip(X, y):
                output = self.forward(xi)
                self.backward(xi, [yi], output, learning_rate)

    # Predict labels for given input data
    def predict(self, X):
        predictions = []
        for x in X:
            output = self.forward(x)
            predictions.append(1 if output[0] > 0.5 else 0)
        return predictions


# Initialize and train the neural network
input_size = len(vocab)
hidden_size = 3
output_size = 1

nn = SimpleNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
print(
    f"Training MLP model on input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}, training_data_size={len(X)}"
)
nn.train(X_train, y_train, epochs=1000, learning_rate=0.1)
print("Training completed.")

# Evaluate the model on the test set
predictions = nn.predict(X_test)
correct = 0
for p, y in zip(predictions, y_test):
    if p == y:
        correct += 1
accuracy = correct / len(y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Test the model with a new sentence
print("Type 'quit' to stop.")
while True:
    new_text = input("> ")
    if new_text.lower() == "quit":
        break
    new_text_vectorized = text_to_vector(new_text)
    prediction = nn.predict([new_text_vectorized])
    print(
        f"Prediction for '{new_text}': {'positive' if prediction[0] == 1 else 'negative'}"
    )
