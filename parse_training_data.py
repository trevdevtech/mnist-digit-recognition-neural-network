import numpy as np
from collections import deque

# parse is responsible for taking the input file
# and breaking into a list of tuples where the first
# element in each tuple is the input activation and the
# second element is the expect output vector
def parse(fo):

	targets = [np.array([[1],[0],[0],[0],[0],[0],[0],[0],[0],[0]]), \
			np.array([[0],[1],[0],[0],[0],[0],[0],[0],[0],[0]]), \
			np.array([[0],[0],[1],[0],[0],[0],[0],[0],[0],[0]]), \
			np.array([[0],[0],[0],[1],[0],[0],[0],[0],[0],[0]]), \
			np.array([[0],[0],[0],[0],[1],[0],[0],[0],[0],[0]]), \
			np.array([[0],[0],[0],[0],[0],[1],[0],[0],[0],[0]]), \
			np.array([[0],[0],[0],[0],[0],[0],[1],[0],[0],[0]]), \
			np.array([[0],[0],[0],[0],[0],[0],[0],[1],[0],[0]]), \
			np.array([[0],[0],[0],[0],[0],[0],[0],[0],[1],[0]]), \
			np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[1]])]

	t_list = []

	activations = []
	actuals = []
	deq = deque()

	for i, line in enumerate(fo):
		line = line.replace("\n", "")
		for j, char in enumerate(line):
			if char == ' ':
				activations.append(np.array(deq))
				deq.clear()
				actuals.append(targets[int(line[j+1])])
				break
			deq.append([float(char)])

	for i, a in enumerate(activations):
		t_list.append((activations[i], actuals[i]))
		if not(len(a) == 1024):
			print("LENGTH is " + str(len(a)))
			print("item # " + str(i))
			exit(-1)

	return t_list
