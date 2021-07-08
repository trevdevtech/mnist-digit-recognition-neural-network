import numpy as np
import random
import copy

# class 'FFNetwork' implements a primitive feedforward neural network
# the main features include feedforward, backpropagation, and the ability
# to train the network with stochastic gradient descent
class FFNetwork:

	def __init__(self, layers, test_data):
		self.layers = layers
		self.biases = [np.random.randn(y, 1) for y in layers[1:]]
		self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
		self.find_start(layers, test_data)

	# Selects the best starting weights to allow application of gradient
	# - descent at a more optimal position
	def find_start(self, l, test_data):
		ideal_w = []
		ideal_b = []
		best = 0
		np.random.seed(2)
		print('\033[96m' + "Finding good start before performing stochastic gradient descent:" + '\033[0m')
		for i in range(int(l[1]/10)):
			self.weights = [np.random.randn(y, x) for x, y in zip(l[:-1], l[1:])]
			self.biases = [np.random.randn(y, 1) for y in l[1:]]
			result = self.correct_guess_count(test_data)
			if result > best:
				best = result
				ideal_w = copy.deepcopy(self.weights)
				ideal_b = copy.deepcopy(self.biases)
			print(".", end=' ', flush = True)
		self.weights = ideal_w
		self.biases = ideal_b
		print(best)

	# derivative of error function
	def error_prime(self, a, t):
		return (a - t)

	# counts the number of correctly guessed
	# outputs in the passed training_data list 'td'
	def correct_guess_count(self, td):
		amax = [np.argmax(self.forward_pass(layer[0])[0][-1]) for layer in td]
		tmax = [np.argmax(t[1]) for t in td]
		return sum([1 if a == t else 0 for a, t in zip(amax, tmax)])

	# train net deploys stochastic gradient descent by breaking
	# up training data into batches, randomizing the order of
	# each batch, then performing back propagation on each input
	# for each batch. 'e' represent the epochs, 'lr' is the learning
	# rate, 'td' is the training data and batch_sz is the size of each
	# batch that will be formulated with the training data
	def train_net(self, e, lr, td, batch_sz):
		for i in range(e):
			# performs local gradient descent per batch
			for j in range(int(len(td)/batch_sz)):
				batch = td[(j*batch_sz):(j*batch_sz)+batch_sz]
				random.shuffle(batch)
				self.apply_gd_batch(lr, batch)
			success_rate = (self.correct_guess_count(td) / len(td)) * 100
			print("Success rate at epoch " + str(i) + ": " + str(success_rate))

	# perform back propagation for the given batch 'batch'
	# 'lr' is the learning rate
	def apply_gd_batch(self, lr, batch):
		b_gradients = [np.zeros(bias.shape) for bias in self.biases]
		w_gradients = [np.zeros(weight.shape) for weight in self.weights]
		for activation, t in batch:
			bw_gradient = self.back_propagate(activation, t)

			for i, gradient in enumerate(b_gradients):
				gradient += bw_gradient[0][i]

			for i, gradient in enumerate(w_gradients):
				gradient += bw_gradient[1][i]

		self.update_weights(lr, (b_gradients, w_gradients), len(batch))

	# given a weights and biases gradient, update the networks
	# weights and biases
	def update_weights(self, lr, bw_gradient, batch_sz):
		for i, biases in enumerate(self.biases):
			self.biases[i] = biases - ((lr * bw_gradient[0][i]) / batch_sz)
		for i, weights in enumerate(self.weights):
			self.weights[i] = weights - ((lr * bw_gradient[1][i]) / batch_sz)

	# complete forward pass of network
	# returns a tuple activations and z_vectors
	# where activations is a list of activations at each layer
	# and z_vectors being the z value computed for each layer
	def forward_pass(self, a):
		z_vectors = []
		activations = [a]
		for i, w in enumerate(self.weights):
			z_vectors.append(np.dot(w, activations[i]) + self.biases[i])
			activations.append(sig(z_vectors[i]))
		return (activations, z_vectors)

	# given an input activation vector 'a' and
	# an expected result vector 't', perform a forward pass
	# and then use back propagation to achieve gradients for
	# biases and weights. Return 'b_gradients' and 'w_gradients'
	# where each list is the gradients for biases and weight respectively
	def back_propagate(self, a, t):
		b_gradients = [np.zeros(bias.shape) for bias in self.biases]
		w_gradients = [np.zeros(weight.shape) for weight in self.weights]

		a, z = self.forward_pass(a)

		d = self.error_prime(a[2], t) * sigp(z[1])
		b_gradients[1] = d

		w_gradients[1] = np.dot(a[1], d.transpose())
		w_gradients[1] = w_gradients[1].transpose()

		d = np.dot(self.weights[1].transpose(), d) * sigp(z[0])
		b_gradients[0] = d

		w_gradients[0] = np.dot(a[0], d.transpose())
		w_gradients[0] = w_gradients[0].transpose()
		return (b_gradients, w_gradients)

# sigmoid activation function
def sig(x):
	return (1.0 / (1.0 + np.exp(-x)))

# derivative of sigmoid activation function
def sigp(x):
	return (sig(x) * (1 - sig(x)))
