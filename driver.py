import sys
from parse_training_data import parse
from ffnetwork import FFNetwork
import numpy as np

# instances the neural network and begins the training process
def train_net(p):
	epochs = int(p[1])
	learn_rate = float(p[2])

	print("opening: " + p[0])
	fo = open(p[0], 'r')
	t_tup = parse(fo)
	fo.close()

	fnet = FFNetwork([1024, 512, 10], t_tup)
	fnet.train_net(epochs, learn_rate, t_tup, 32)

def main():
	parameters = []

	if len(sys.argv) < 2:
		print("please specifiy training data set")
		exit(-1)
	else:
		parameters.append(sys.argv[1])

		if len(sys.argv) < 3:
			print("No epoch amount specified, defaulting to 18")
			parameters.append("18")
		else:
			parameters.append(sys.argv[2])

		if len(sys.argv) < 4:
			print("No learning rate specified, defaulting to 2")
			parameters.append("2")
		else:
			parameters.append(sys.argv[3])

		train_net(parameters)

main()

