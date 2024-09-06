import pandas as pd
import numpy as np

# Task 1
# 1A)
print("\n Task 1A\n")

# Read the full data file
full_data = pd.read_csv("SpotifyFeatures.csv", sep=",")

# Print out the length of the genre list
# Since the full data is a nxm matrix where n is the song count we just use that
print("Number of songs: ", full_data.shape[0])


# Print out the number of features
# Since the full data is a nxm matrix where m is the feature count we just use that
print("Number of features: ", full_data.shape[1])


# 1B)
print("\n Task 1B\n")

# Keep only the columns we care about
data_set = full_data[["genre", "loudness", "liveness"]]


# Filter out Pop songs
pop_songs = data_set[data_set["genre"] == "Pop"].copy()
pop_songs["label"] = 1  # Assign label 1 for Pop


# Filter out Classical songs
classic_songs = data_set[data_set["genre"] == "Classical"].copy()
classic_songs["label"] = 0  # Assign label 0 for Classical

# Print number of Pop and Classical songs
print("Number of pop songs: ", len(pop_songs))
print("Number of classical songs: ", len(classic_songs))


# 1C)

from sklearn.model_selection import train_test_split


# Split the classical songs into a training set and test set
classic_train, classic_test = train_test_split(classic_songs, test_size=0.2, random_state=42)

# Split the pop songs into a training set and test set
pop_train, pop_test = train_test_split(pop_songs, test_size=0.2, random_state=42)

# Combine the training and test sets
train_set = pd.concat([pop_train, classic_train])
test_set = pd.concat([pop_test, classic_test])

# Shuffle the datasets
train_set = train_set.sample(frac=1, random_state=42).reset_index(drop=True)
test_set = test_set.sample(frac=1, random_state=42).reset_index(drop=True)


# Extract features and labels
train_features = train_set[["loudness", "liveness"]].values
train_labels = train_set["label"].values.reshape(-1, 1)
test_features = test_set[["loudness", "liveness"]].values
test_labels = test_set["label"].values.reshape(-1, 1)



# 1D)

import matplotlib.pyplot as plt
print("\n Task 1D\n")

# Plot the data
# Pop training
plt.scatter(train_features[train_labels.flatten() == 1, 0],
            train_features[train_labels.flatten() == 1, 1],
            color='blue', s=4, label='Pop train')
# Pop test
plt.scatter(test_features[test_labels.flatten() == 1, 0],
            test_features[test_labels.flatten() == 1, 1],
            color='purple', s=4, label='Pop test')
# Classical training
plt.scatter(train_features[train_labels.flatten() == 0, 0],
            train_features[train_labels.flatten() == 0, 1],
            color='red', s=4, label='Classical train')
# Classical test
plt.scatter(test_features[test_labels.flatten() == 0, 0],
            test_features[test_labels.flatten() == 0, 1],
            color='orange', s=4, label='Classical test')



plt.title("Loudness and Liveness")
plt.xlabel("Loudness")
plt.ylabel("Liveness")
plt.legend()
plt.show()


# 2
# 2A)
print("\nTask 2A) \n")

# Defining the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Defining SGD logistic regression function
def model(features, labels, learning_rate, iterations):
    # Initializing values
    rows = features.shape[0]
    cols = features.shape[1]
    w = np.zeros(cols)  # Initialize weights as an array of zeros
    b = 0.0  # Initialize bias
    cost_list = [] # An empty list used for plotting later

    # Looping over the following calulations {iteration} times
    for iteration in range(iterations):
        total_cost = 0 # Reset total cost every loop

        # Forward and backward propagation for all samples
        for sample in range(rows):  # Loop over each sample

            # Select a single sample, both a feature and label
            feature = features[sample]
            label = labels[sample]

            # Calculation of the linear value and sigmoid function
            linear = np.dot(feature, w) + b
            prediction = sigmoid(linear)

            # Backward propagation for a single sample
            dw = (prediction - label) * feature
            db = prediction - label

            # Update weight and bias
            w -= learning_rate * dw
            b -= learning_rate * db

            # Calculate the cost/loss for this sample
            cost = -(label * np.log(prediction) + (1 - label) * np.log(1 - prediction))
            total_cost += cost # Add to total

        # Average cost over all samples
        avg_cost = total_cost / rows
        cost_list.append(avg_cost)
        
        # While working all this out we want to get some sort of update to see that the code works. So this will return an update of the current cost 10 times while the code is running
        if iteration % (iterations // 10) == 0:
            print(f"Cost after iteration {iteration}: {avg_cost}")
            
    # Returning the stuff we care about
    return w, b, cost_list

# Setting parameters
iterations = 10
learning_rate = 0.001

# Train the model
w, b, cost_list = model(train_features, train_labels, learning_rate, iterations)


# Plotting cost function
plt.title(f"Iterations={iterations} Learning rate={learning_rate}")
plt.plot(np.arange(iterations), cost_list)
plt.grid()
plt.show()


# Importing accuracy score calculator
from sklearn.metrics import accuracy_score


# Predictor
def pred(x):
    # Calculating the Z for the input using the weights and bias from the training we did
     linear_modell = np.dot(x, w) + b
     
     # Making a prediction for the labels
     y_pred = sigmoid(linear_modell)

     # Looking at the predictions we made, and making it a 1 if it is equal to or above 0.5, else it's a 0
     y_pred = np.where(y_pred >= 0.5, 1 ,0)
     
     return y_pred

# Making the labels of the training set into an array, and flattening it
train_labels = np.array(train_labels.flatten(), dtype= np.int64)

# Making predicted labels using the prediction function we made
train_labels_hat = pred(train_features)

# Using the accuracy score function from sklearn to find a % value
print(f"Accuracy score on training set: {round(accuracy_score(train_labels, train_labels_hat)*100,2)}%")

#2B

print("\n Task 2B) \n")

# Making the labels of the test set into an array, and flattening it
test_labels = np.array(test_labels.flatten(), dtype= np.int64)

# Making the predicted labels using the prediction function from earlier
test_labels_hat = pred(test_features)

# Using the accuracy score function from sklearn to find a % value
print(f"Accuracy score on test set: {round(accuracy_score(test_labels, test_labels_hat)*100,2)}%")

#3A
print("\n Task 3A) \n")
from sklearn.metrics import confusion_matrix

# Printing out the confusion matrix
print("Confusion matrix:\n",confusion_matrix(test_labels, test_labels_hat))