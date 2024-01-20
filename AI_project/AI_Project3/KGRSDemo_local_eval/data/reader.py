import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.metrics import accuracy_score



class MatrixFactorization:
    def __init__(self, num_users, num_items, num_factors):
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.P = np.random.normal(scale=1./self.num_factors, size=(self.num_users, self.num_factors))
        self.Q = np.random.normal(scale=1./self.num_factors, size=(self.num_items, self.num_factors))

    def train(self, train_pos, train_neg, iterations, learning_rate, neg_sample_size):
        self.loss_history = []
        # Iterate over the number of iterations
        for it in range(iterations):
            # Shuffle the training data for stochastic gradient descent
            np.random.shuffle(train_pos)

            # Update embeddings using positive samples
            for u, i, r in train_pos:
                error_ui = r - np.dot(self.P[u, :], self.Q[i, :].T)
                self.P[u, :] += learning_rate * error_ui * self.Q[i, :]
                self.Q[i, :] += learning_rate * error_ui * self.P[u, :]

            # Select a subset of negative samples
            neg_sample_indices = np.random.choice(len(train_neg), neg_sample_size, replace=False)
            for idx in neg_sample_indices:
                u, i, r = train_neg[idx]
                error_ui = r - np.dot(self.P[u, :], self.Q[i, :].T)
                self.P[u, :] += learning_rate * error_ui * self.Q[i, :]
                self.Q[i, :] += learning_rate * error_ui * self.P[u, :]

            # Optionally, compute and print the loss at each iteration
            loss = self.compute_loss(train_pos)
            self.loss_history.append(loss)
            print(f"Iteration {it+1}/{iterations}, loss: {loss}")

    def compute_loss(self, train_pos):
        
        loss = 0
        for u, i, r in train_pos:
            #error_ui = abs(r - np.dot(self.P[u, :], self.Q[i, :].T))
            error_ui = (r - np.dot(self.P[u, :], self.Q[i, :].T)) ** 2
            loss += error_ui
        return loss

    def predict(self, user_id, item_id):
        return np.dot(self.P[user_id], self.Q[item_id].T)
    


neg_data = np.load('train_neg.npy')
pos_data = np.load('train_pos.npy')

# print('pos data: \n')
# print(pos_data)
# print('neg data: \n')
# print(neg_data)

num_users = max(np.max(pos_data[:, 0]), np.max(neg_data[:, 0])) + 1
num_items = max(np.max(pos_data[:, 1]), np.max(neg_data[:, 1])) + 1

print('num_users: ' + str(num_users))
print('num_items: ' + str(num_items))

'''
num_users: 6015
num_items: 2347

'''

d = 10

mf_model = MatrixFactorization(num_users, num_items, d)

# # Train the model
mf_model.train(pos_data, neg_data, iterations=100, learning_rate=0.01, neg_sample_size=100)

# P_matrix = mf_model.P
# Q_matrix = mf_model.Q

# print('matrix P: ')
# print(P_matrix)
# print('matrix Q: ')
# print(Q_matrix)
user_id, item_id = 6014, 1485 
prediction = mf_model.predict(user_id, item_id)
print('prediction user '+ str(user_id) + ' item ' + str(item_id) + ' : ' + str(prediction))

all_data = np.vstack((pos_data, neg_data))
train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)

# Define predict_interaction function
# def predict_interaction(user_id, item_id, model):
#     return model.predict(user_id, item_id)

# Generate predictions for test_data
predictions = [mf_model.predict(user, item) for user, item, _ in test_data]
actuals = [actual_interaction for _, _, actual_interaction in test_data]

new_threshold = 0.5  

# Convert predictions to binary values based on the new threshold
binary_predictions = [1 if pred >= new_threshold else 0 for pred in predictions]

accuracy = accuracy_score(actuals, binary_predictions)
accuracy_history = [accuracy]
print(f"Accuracy: {accuracy}")

# Calculate classification metrics
precision = precision_score(actuals, binary_predictions, zero_division=0)
recall = recall_score(actuals, binary_predictions, zero_division=0)
f1 = f1_score(actuals, binary_predictions, zero_division=0)

# For rating prediction metrics
rmse = sqrt(mean_squared_error(actuals, predictions))
mae = mean_absolute_error(actuals, predictions)

print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}, RMSE: {rmse}, MAE: {mae}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(mf_model.loss_history)
plt.title('Training Loss Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss')

# Plotting the accuracy
plt.subplot(1, 2, 2)
plt.bar(['Accuracy'], accuracy_history)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.savefig('graph.png')





