from practice_10.utils import sigmoid
import numpy as np


class ANNModel:
    def __init__(self, n_x, n_h=4, n_y=1, learning_rate=1.2, random_seed=42):
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y
        self.learning_rate = learning_rate

        self.parameters = {}
        self.cache = {}
        self.costs = []

        np.random.seed(random_seed)
        self._initialize_parameters()

    def _initialize_parameters(self):
        layer_dims = [(self.n_x, self.n_h), (self.n_h, self.n_y)]

        for idx, (in_dim, out_dim) in enumerate(layer_dims, 1):
            self.parameters[f'W{idx}'] = np.random.randn(out_dim, in_dim) * 0.01
            self.parameters[f'b{idx}'] = np.zeros((out_dim, 1))

    def _forward_propagation(self, X):
        W1 = self.parameters['W1']
        b1 = self.parameters['b1']
        W2 = self.parameters['W2']
        b2 = self.parameters['b2']

        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)

        self.cache = {
            'Z1': Z1,
            'A1': A1,
            'Z2': Z2,
            'A2': A2
        }

        return A2

    @staticmethod
    def _compute_cost(A2, Y):
        m = Y.shape[1]
        logprobs = np.dot(Y, np.log(A2).T) + np.dot(1 - Y, np.log(1 - A2).T)
        cost = np.float64(-logprobs / m)
        return np.squeeze(cost)

    def _backward_propagation(self, X, Y):
        m = X.shape[1]

        W2 = self.parameters['W2']
        A1 = self.cache['A1']
        A2 = self.cache['A2']

        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        grads = {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2
        }

        return grads

    def _update_parameters(self, grads):
        for idx in [1, 2]:
            self.parameters[f'W{idx}'] -= self.learning_rate * grads[f'dW{idx}']
            self.parameters[f'b{idx}'] -= self.learning_rate * grads[f'db{idx}']

    def fit(self, X, Y, num_iterations=10000, print_cost=False):
        self.costs = []

        for i in range(num_iterations):
            A2 = self._forward_propagation(X)
            cost = self._compute_cost(A2, Y)
            grads = self._backward_propagation(X, Y)
            self._update_parameters(grads)

            self.costs.append(cost)

            if print_cost and i % 1000 == 0:
                print(f"Cost after iteration {i}: {cost:.6f}")

        return self

    def predict_proba(self, X):
        return self._forward_propagation(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)

    def score(self, X, Y):
        predictions = self.predict(X)
        Y_flat = Y.ravel()
        predictions_flat = predictions.ravel()
        accuracy = np.mean(predictions_flat == Y_flat) * 100
        return accuracy

    def get_decision_boundary_function(self):
        def boundary_func(x):
            return self.predict(x.T).T
        return boundary_func
