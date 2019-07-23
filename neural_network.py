"""
date : 23 July 2019
author : MengXiangzhe
description: a neural network practice, coding a neural network with training and test process,
            which is used for numeral recognition
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.special


class neuralnetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.activation_function = lambda x: scipy.special.expit(x)
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5),(self.onodes, self.hnodes))
        self.lr = learningrate

    def train(self, input_list, targets_list):
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        self.who += self.lr * np.dot((output_errors * final_outputs *(1 - final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1- hidden_outputs)), np.transpose(inputs))

    def query(self,inputs_list):
        inputs = np.array(inputs_list, ndmin = 2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_output = self.activation_function(final_inputs)
        return final_output


def main():
    # training neural network:
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.3
    n = neuralnetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    training_data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    for record in training_data_list:
        all_values = record.split(',')
        input = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(input, targets)

    #testing neural network:
    test_data_file = open("mnist_dataset/mnist_test_10.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    res_list=[]
    for record in test_data_list:
        all_test_values = record.split(',')
        input_list = (np.asfarray(all_test_values[1:]) / 255.0 * 0.99) + 0.01
        #print(np.argmax(np.copy(n.query(input_list))))
        if np.argmax(np.copy(n.query(input_list))) == int(all_test_values[0]):
            res_list.append(1)
        else:
            res_list.append(0)
        plt.imshow(np.asfarray(all_test_values[1:]).reshape((28,28)))
        print(np.argmax(np.copy(n.query(input_list))))
        plt.pause(1)
    res_array = np.asarray(res_list)
    print("performance =  ", res_array.sum() / res_array.size)

if __name__ == '__main__':
    main()

