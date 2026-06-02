import numpy as np
import copy

class FeedforwardNeuralNetwork:
    def __init__(
        self,
        num_layers,
        num_features,
        num_hidden_units: list,
        num_classes,
        activation_func,
        activation_func_prime,
        weights=None,
    ):
        self.activation_func = activation_func
        self.activation_func_prime = activation_func_prime
        self.num_layers = num_layers

        self.num_features = num_features
        self.num_hidden_units = num_hidden_units
        self.num_classes = num_classes

        if weights is None:
            self.weights = self.initialize_weights()
        else:
            self.weights = weights

        self.output = None

    def he_initialization(self, l_idx):
        current_l_node_num = self.get_layer_size(l_idx)
        w = np.random.randn(self.get_layer_size(l_idx + 1), current_l_node_num + 1) * np.sqrt(
            2 / current_l_node_num
        )
        return w

    def xavier_init(self, l_idx):
        current_l_node_num = self.get_layer_size(l_idx)
        next_l_node_num = self.get_layer_size(l_idx + 1)

        limit = np.sqrt(6 / (current_l_node_num + next_l_node_num))
        w = np.random.uniform(-limit, limit, (next_l_node_num, current_l_node_num + 1))
        return w

    def initialize_weights(self):
        weights = []

        for i in range(self.num_layers - 1):
            if i == self.num_layers - 2:
                w = self.xavier_init(i)
            else:
                w = self.he_initialization(i)
            weights.append(w)

        return weights

    def get_layer_size(self, layer_idx):
        if layer_idx == 0:
            return self.num_features
        elif layer_idx == self.num_layers - 1:
            return self.num_classes
        elif 0 < layer_idx < self.num_layers - 1:
            return self.num_hidden_units[layer_idx - 1]
        else:
            raise IndexError

    def forward_pass(self, x_data, weights):
        z_list = []
        activations_list = []

        for layer_idx in range(self.num_layers):
            if layer_idx == 0:
                z_list.append(None)
                activations_list.append(x_data)
            else:
                pre_activation_no_bias = activations_list[layer_idx - 1]
                pre_activation = np.hstack(
                    [np.ones((pre_activation_no_bias.shape[0], 1)), pre_activation_no_bias]
                )

                curr_z = pre_activation.dot(weights[layer_idx - 1].T)
                z_list.append(curr_z)

                curr_activation = self.activation_func[layer_idx - 1](curr_z)
                activations_list.append(curr_activation)

        return activations_list, z_list

    def l2_regularization(self, lambda_, weights, m):
        regularization = 0
        for weight in weights:
            squared_weights = weight[:, 1:] ** 2
            regularization += np.sum(squared_weights)

        regularization = (lambda_ / (2 * m)) * regularization
        return regularization

    def binary_cross_entropy_loss(self, h_theta, y_data, weights, regularize=True, lambda_=0):
        m = y_data.shape[0]

        eps = 1e-12
        h = np.clip(h_theta, eps, 1 - eps)

        cost = -np.sum(y_data * np.log(h)+ (1 - y_data) * np.log(1 - h)) / m

        if regularize:
            cost += self.l2_regularization(lambda_, weights, m)

        return cost

    def backpropagation(self, y_data, activations, z_list, weights, lambda_=0):
        g_prime = self.activation_func_prime
        num_layers = self.num_layers

        delta = [None] * num_layers
        delta[num_layers - 1] = activations[-1] - y_data

        for layer_idx in range(num_layers - 2, 0, -1):
            theta_l = weights[layer_idx]

            delta[layer_idx] = (
                delta[layer_idx + 1].dot(theta_l[:, 1:])
                * g_prime[layer_idx - 1](z_list[layer_idx])
            )

        m = y_data.shape[0]
        gradients = [None] * (num_layers - 1)

        for layer_idx in range(num_layers - 1):
            gradients[layer_idx] = (delta[layer_idx + 1].T.dot(np.hstack((np.ones((m, 1)), activations[layer_idx]))) / m)

            gradients[layer_idx][:, 1:] += (lambda_ / m) * weights[layer_idx][:, 1:]

        return gradients

    def gradient_check(self, x_data, y_data, gradients, weights):
        epsilon = 1e-5

        for layer_idx in range(len(weights)):
            num_params = 10

            for _ in range(num_params):
                i = np.random.randint(0, weights[layer_idx].shape[0])
                j = np.random.randint(0, weights[layer_idx].shape[1])

                theta_plus = copy.deepcopy(weights)
                theta_minus = copy.deepcopy(weights)

                theta_plus[layer_idx][i, j] += epsilon
                theta_minus[layer_idx][i, j] -= epsilon

                h_plus, _ = self.forward_pass(x_data, theta_plus)
                h_minus, _ = self.forward_pass(x_data, theta_minus)

                grad_approx = (
                    self.binary_cross_entropy_loss(h_plus[-1], y_data, theta_plus)
                    - self.binary_cross_entropy_loss(h_minus[-1], y_data, theta_minus)
                ) / (2 * epsilon)

                diff = abs(grad_approx - gradients[layer_idx][i, j])

                if diff > 1e-3:
                    print((i, j), "at Layer", layer_idx, "FAILED. diff =", diff)
                    return False

        return True

    def update_weights(self, gradients, weights, learning_rate):
        for layer_idx in range(len(weights)):
            weights[layer_idx] -= learning_rate * gradients[layer_idx]

        return weights

    def fit(self, x_data, y_data, num_epoch, learning_rate, lambda_=0, batch_size = 32, debug=False):
        num_batch = max(1, len(x_data) // batch_size)
        training_epochs_cost = []

        for epoch in range(num_epoch):
            idx = np.random.permutation(len(x_data))
            x_data = x_data[idx]
            y_data = y_data[idx]

            x_batches = np.array_split(x_data, num_batch)
            y_batches = np.array_split(y_data, num_batch)
                
            for i in range(num_batch):

                x_batch = x_batches[i]
                y_batch = y_batches[i]

                activations, z_list = self.forward_pass(x_batch, self.weights)

                gradients = self.backpropagation(
                    y_batch, activations, z_list, self.weights, lambda_
                )

                if epoch == 0 and debug:
                    flag = self.gradient_check(
                        x_batch, y_batch, gradients, self.weights
                    )
                    if not flag:
                        raise RuntimeError

                self.weights = self.update_weights(
                    gradients, self.weights, learning_rate
                )

            full_activations, _ = self.forward_pass(x_data, self.weights)

            cost = self.binary_cross_entropy_loss(
                full_activations[-1], y_data, self.weights, True, lambda_
            )

            training_epochs_cost.append(cost)

        return training_epochs_cost

    def predict_classes(self, x_data):
        activations, _ = self.forward_pass(x_data, self.weights)
        self.output = (activations[-1] > 0.5).astype(int)
        return self.output
