#Author: Grant Baker
import numpy as np
import matplotlib.pyplot as plt
#perform experiments
def main():
    training_data = read_training_data("train-images-idx3-ubyte")
    training_data = np.divide(training_data, 255)
    training_label = read_training_label("train-labels-idx1-ubyte")
    test_data = read_test_data("t10k-images-idx3-ubyte")
    test_data = np.divide(test_data, 255)
    test_label = read_test_label("t10k-labels-idx1-ubyte")
    experiment1(training_data, training_label, test_data, test_label)
    experiment2(training_data, training_label, test_data, test_label)
    experiment3(training_data, training_label, test_data, test_label)
def experiment1(training_data, training_label, test_data, test_label):
    hidden_units = 20
    input_weights = np.random.uniform(low=-.05, high=.05, size=(hidden_units,784))
    input_bias = np.random.uniform(low=-.05, high=.05, size=(hidden_units))
    hidden_weights = np.random.uniform(low=-.05, high=.05, size=(10,hidden_units))
    hidden_bias = np.random.uniform(low=-.05, high=.05, size=(10))
    momentum = 0.9
    learning_rate = 0.1
    x = np.linspace(0,50,51)
    training_20_nodes = np.zeros(51)
    training_50_nodes = np.zeros(51)
    training_100_nodes = np.zeros(51)
    test_20_nodes = np.zeros(51)
    test_50_nodes = np.zeros(51)
    test_100_nodes = np.zeros(51)
    #train model with 20 nodes
    for i in range(51):
        permutation = np.random.permutation(training_data.shape[0])
        training_data = training_data[permutation]
        training_label = training_label[permutation]
        training_20_nodes[i] = network_accuracy(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, training_data, training_label)
        test_20_nodes[i] = network_accuracy(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, test_data, test_label)
        if (i == 50):
            break
        train_network(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, learning_rate, momentum, training_data, training_label)
    matrix_1 = confusion_matrix(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, test_data, test_label)
    hidden_units = 50
    input_weights = np.random.uniform(low=-.05, high=.05, size=(hidden_units,784))
    input_bias = np.random.uniform(low=-.05, high=.05, size=(hidden_units))
    hidden_weights = np.random.uniform(low=-.05, high=.05, size=(10,hidden_units))
    hidden_bias = np.random.uniform(low=-.05, high=.05, size=(10))
    #train model with 50 nodes
    for i in range(51):
        permutation = np.random.permutation(training_data.shape[0])
        training_data = training_data[permutation]
        training_label = training_label[permutation]
        training_50_nodes[i] = network_accuracy(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, training_data, training_label)
        test_50_nodes[i] = network_accuracy(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, test_data, test_label)
        if (i == 50):
            break
        train_network(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, learning_rate, momentum, training_data, training_label)
    matrix_2 = confusion_matrix(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, test_data, test_label)
    
    hidden_units = 100
    input_weights = np.random.uniform(low=-.05, high=.05, size=(hidden_units,784))
    input_bias = np.random.uniform(low=-.05, high=.05, size=(hidden_units))
    hidden_weights = np.random.uniform(low=-.05, high=.05, size=(10,hidden_units))
    hidden_bias = np.random.uniform(low=-.05, high=.05, size=(10))
    #train model with 100 nodes
    for i in range(51):
        permutation = np.random.permutation(training_data.shape[0])
        training_data = training_data[permutation]
        training_label = training_label[permutation]
        training_100_nodes[i] = network_accuracy(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, training_data, training_label)
        test_100_nodes[i] = network_accuracy(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, test_data, test_label)
        if (i == 50):
            break
        train_network(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, learning_rate, momentum, training_data, training_label)
    matrix_3 = confusion_matrix(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, test_data, test_label)
    #print confusion matricies
    print("20 nodes:")
    print(matrix_1)
    print("50 nodes:")
    print(matrix_2)
    print("100 nodes:")
    print(matrix_3)
    #print graph
    plt.figure()
    plt.axis([0,50,0,1])
    plt.plot(x, training_20_nodes, 'r', label='20 nodes')
    plt.plot(x,test_20_nodes, 'r--')
    plt.plot(x, training_50_nodes, 'g', label='50 nodes')
    plt.plot(x,test_50_nodes, 'g--')
    plt.plot(x,training_100_nodes, 'b', label='100 nodes')
    plt.plot(x, test_100_nodes, 'b--')
    plt.legend()
    plt.show()

def experiment2(training_data, training_label, test_data, test_label):
    hidden_units = 100
    input_weights = np.random.uniform(low=-.05, high=.05, size=(hidden_units,784))
    input_bias = np.random.uniform(low=-.05, high=.05, size=(hidden_units))
    hidden_weights = np.random.uniform(low=-.05, high=.05, size=(10,hidden_units))
    hidden_bias = np.random.uniform(low=-.05, high=.05, size=(10))
    momentum = 0
    learning_rate = 0.1
    x = np.linspace(0,50,51)
    training_0_momentum = np.zeros(51)
    training_25_momentum = np.zeros(51)
    training_50_momentum = np.zeros(51)
    test_0_momentum = np.zeros(51)
    test_25_momentum = np.zeros(51)
    test_50_momentum = np.zeros(51)
    #train model with 0 momentum
    for i in range(51):
        permutation = np.random.permutation(training_data.shape[0])
        training_data = training_data[permutation]
        training_label = training_label[permutation]
        training_0_momentum[i] = network_accuracy(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, training_data, training_label)
        test_0_momentum[i] = network_accuracy(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, test_data, test_label)
        if (i == 50):
            break
        train_network(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, learning_rate, momentum, training_data, training_label)
    matrix_1 = confusion_matrix(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, test_data, test_label)

    momentum = 0.25
    input_weights = np.random.uniform(low=-.05, high=.05, size=(hidden_units,784))
    input_bias = np.random.uniform(low=-.05, high=.05, size=(hidden_units))
    hidden_weights = np.random.uniform(low=-.05, high=.05, size=(10,hidden_units))
    hidden_bias = np.random.uniform(low=-.05, high=.05, size=(10))
    #train model with momentum = 0.25
    for i in range(51):
        permutation = np.random.permutation(training_data.shape[0])
        training_data = training_data[permutation]
        training_label = training_label[permutation]
        training_25_momentum[i] = network_accuracy(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, training_data, training_label)
        test_25_momentum[i] = network_accuracy(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, test_data, test_label)
        if (i == 50):
            break
        train_network(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, learning_rate, momentum, training_data, training_label)
    matrix_2 = confusion_matrix(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, test_data, test_label)

    momentum = 0.5
    input_weights = np.random.uniform(low=-.05, high=.05, size=(hidden_units,784))
    input_bias = np.random.uniform(low=-.05, high=.05, size=(hidden_units))
    hidden_weights = np.random.uniform(low=-.05, high=.05, size=(10,hidden_units))
    hidden_bias = np.random.uniform(low=-.05, high=.05, size=(10))
    #train model with momentum = 0.5
    for i in range(51):
        permutation = np.random.permutation(training_data.shape[0])
        training_data = training_data[permutation]
        training_label = training_label[permutation]
        training_50_momentum[i] = network_accuracy(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, training_data, training_label)
        test_50_momentum[i] = network_accuracy(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, test_data, test_label)
        if (i == 50):
            break
        train_network(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, learning_rate, momentum, training_data, training_label)
    matrix_3 = confusion_matrix(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, test_data, test_label)
    #print confusion matricies
    print("Momentum 0:")
    print(matrix_1)
    print("Momentum 0.25:")
    print(matrix_2)
    print("Momentum 0.5:")
    print(matrix_3)
    #print graph
    plt.figure()
    plt.axis([0,50,0,1])
    plt.plot(x, training_0_momentum, 'r', label='Momentum = 0')
    plt.plot(x,test_0_momentum, 'r--')
    plt.plot(x, training_25_momentum, 'g', label='Momentum = 0.25')
    plt.plot(x,test_25_momentum, 'g--')
    plt.plot(x,training_50_momentum, 'b', label='Momentum = 0.5')
    plt.plot(x, test_50_momentum, 'b--')
    plt.legend()
    plt.show()
def experiment3(training_data, training_label, test_data, test_label):
    hidden_units = 100
    input_weights = np.random.uniform(low=-.05, high=.05, size=(hidden_units,784))
    input_bias = np.random.uniform(low=-.05, high=.05, size=(hidden_units))
    hidden_weights = np.random.uniform(low=-.05, high=.05, size=(10,hidden_units))
    hidden_bias = np.random.uniform(low=-.05, high=.05, size=(10))
    momentum = 0.9
    learning_rate = 0.1
    x = np.linspace(0,50,51)
    traininghalfdata = np.zeros(51)
    trainingquarterdata = np.zeros(51)
    testhalfdata = np.zeros(51)
    testquarterdata = np.zeros(51)
    #randomize data
    permutation = np.random.permutation(training_data.shape[0])
    training_data = training_data[permutation]
    training_label = training_label[permutation]
    halfdata = training_data[0:30000]
    halflabel = training_label[0:30000]
    quarterdata = training_data[30000:45000]
    quarterlabel = training_label[30000:45000]
    #train model with half the data
    for i in range(51):
        permutation = np.random.permutation(halfdata.shape[0])
        halfdata = halfdata[permutation]
        halflabel =halflabel[permutation]
        traininghalfdata[i] = network_accuracy(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, halfdata, halflabel)
        testhalfdata[i] = network_accuracy(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, test_data, test_label)
        if (i == 50):
            break
        train_network(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, learning_rate, momentum, halfdata, halflabel)
    matrix_1 = confusion_matrix(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, test_data, test_label)

    input_weights = np.random.uniform(low=-.05, high=.05, size=(hidden_units,784))
    input_bias = np.random.uniform(low=-.05, high=.05, size=(hidden_units))
    hidden_weights = np.random.uniform(low=-.05, high=.05, size=(10,hidden_units))
    hidden_bias = np.random.uniform(low=-.05, high=.05, size=(10))
    #train model with quarter of the data
    for i in range(51):
        permutation = np.random.permutation(quarterdata.shape[0])
        quarterdata = quarterdata[permutation]
        quarterlabel = quarterlabel[permutation]
        trainingquarterdata[i] = network_accuracy(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, quarterdata, quarterlabel)
        testquarterdata[i] = network_accuracy(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, test_data, test_label)
        if (i == 50):
            break
        train_network(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, learning_rate, momentum, quarterdata, quarterlabel)
    matrix_2 = confusion_matrix(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, test_data, test_label)

    #print confusion matrices
    print("Half Data")
    print(matrix_1)
    print("Quarter Data")
    print(matrix_2)
    #print graph
    plt.figure()
    plt.axis([0,50,0,1])
    plt.plot(x, traininghalfdata, 'r', label='Half Data')
    plt.plot(x,testhalfdata, 'r--')
    plt.plot(x, trainingquarterdata, 'g', label='Quarter Data')
    plt.plot(x,testquarterdata, 'g--')
    plt.legend()
    plt.show()
#calculate accuracy of model over dataset
def network_accuracy(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, training_data, training_label):
    count = 0
    hidden_values = np.zeros(hidden_units)
    output_values = np.zeros(10)
    data_points = training_data.shape[0]
    for i in range(data_points):
        hidden_values = activation(np.add(np.matmul(input_weights, training_data[i,:]), input_bias))
        output_values = activation(np.add(np.matmul(hidden_weights, hidden_values), hidden_bias))
        actual_value = training_label[i]
        prediction = np.argmax(output_values)
        if (actual_value == prediction):
            count = count + 1
    return count / data_points
#calculate confusion matrix of model over dataset
def confusion_matrix(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, training_data, training_label):
    matrix = np.zeros((10,10))
    hidden_values = np.zeros(hidden_units)
    output_values = np.zeros(10)
    data_points = training_data.shape[0]
    for i in range(data_points):
        hidden_values = activation(np.add(np.matmul(input_weights, training_data[i,:]), input_bias))
        output_values = activation(np.add(np.matmul(hidden_weights, hidden_values), hidden_bias))
        actual_value = training_label[i]
        prediction = np.argmax(output_values)
        matrix[actual_value, prediction] += 1
    return matrix
#trains neural network for one epoch
#input_weights and hidden_weights are weight matrices
#The ith row and jth column of the input weights is the weight from the ith data point to the jth hidden node
#The ith row and jth column of the hidden weights is the weight from the ith hidden node to the jth output node
#input_bias and hidden_bias is a weight vector where the ith component is the weight to the ith hidden node and weight to the ith output node respectively
#The hidden_units, learning_rate, and momentum are model parameters (constants)
#The training data is a matrix and the training_label is a vector where the label of the ith entry in the training_label corresponds to
#the data of the ith row of the training_data
#The following neural network with one hidden layer is done via backpropogation
def train_network(input_weights, input_bias, hidden_weights, hidden_bias, hidden_units, learning_rate, momentum, training_data, training_label):
    input_error_change = np.zeros((hidden_units, 784)) #change of input weights
    input_bias_change = np.zeros(hidden_units) #change of input bias
    hidden_error_change = np.zeros((10, hidden_units)) #change of hidden weights
    hidden_bias_change = np.zeros(10) #change of hidden bias
    hidden_values = np.zeros(hidden_units)
    output_values = np.zeros(10)
    data_points = training_data.shape[0]
    expected_output = np.zeros(10)
    for i in range(data_points):
        hidden_values = activation(np.add(np.matmul(input_weights, training_data[i,:]), input_bias))
        output_values = activation(np.add(np.matmul(hidden_weights, hidden_values), hidden_bias))
        actual_value = training_label[i]
        prediction = np.argmax(output_values)
        for j in range(10):
            if (j == actual_value):
                expected_output[j] = 0.9
            else:
                expected_output[j] = 0.1
        #skip backprop if neural network succesfully predicts label
        if (actual_value == prediction):
            continue
        #backprop algorithm
        output_error = output_values * (1 - output_values) * (expected_output - output_values)
        hidden_error = hidden_values * (1- hidden_values) * np.matmul(np.transpose(hidden_weights), output_error)
        hidden_error_change = np.multiply(learning_rate, np.outer(output_error, hidden_values)) + np.multiply(momentum, hidden_error_change)
        hidden_bias_change = np.multiply(learning_rate, output_error) + np.multiply(momentum, hidden_bias_change)
        input_error_change = np.multiply(learning_rate, np.outer(hidden_error, training_data[i,:])) + np.multiply(momentum, input_error_change)
        input_bias_change = np.multiply(learning_rate, hidden_error) + np.multiply(momentum, input_bias_change)
        input_weights += input_error_change
        input_bias += input_bias_change
        hidden_weights += hidden_error_change
        hidden_bias += hidden_bias_change




#sigmoid activation function
def activation(input):
    temp = np.exp(-input)
    temp = np.add(temp, 1)
    return np.divide(1, temp)
#read in data and labels from files
def read_training_data(filename):
    file = open(filename, 'rb')
    file.read(16)
    shape = (60000, 784)
    return np.frombuffer(file.read(), dtype=np.uint8).reshape(shape)
def read_test_data(filename):
    file = open(filename, 'rb')
    file.read(16)
    shape = (10000, 784)
    return np.frombuffer(file.read(), dtype=np.uint8).reshape(shape)
def read_training_label(filename):
    file = open(filename, 'rb')
    file.read(8)
    shape = (60000,1)
    return np.frombuffer(file.read(), dtype=np.uint8).reshape(shape)
def read_test_label(filename):
    file = open(filename, 'rb')
    file.read(8)
    shape = (10000,1)
    return np.frombuffer(file.read(), dtype=np.uint8).reshape(shape)


if __name__ == "__main__":
    main()