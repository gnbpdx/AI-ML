import numpy as np
import csv
def main():
    data = read_data("data/spambase.data")
    (rows, cols) = data.shape
    labels = data[:, cols - 1]
    data = data[:, 0: cols - 1]
    cols -= 1
    permutation = np.random.permutation(rows)
    data = data[permutation]
    labels = labels[permutation]
    training_data = data[0:2300, :]
    test_data = data[2300:,:]
    training_labels = labels[0:2300]
    test_labels = labels[2300:]
    prior_spam = compute_prior_spam(training_labels)
    prior_not_spam= 1 - prior_spam
    means_spam = np.mean(training_data[training_labels[:] == 1, :], axis=0)
    means_not_spam = np.mean(training_data[training_labels[:] == 0, :], axis=0)
    stds_spam = np.std(training_data[training_labels[:] == 1, :], axis=0)
    stds_not_spam = np.std(training_data[training_labels[:] == 0, :], axis=0)
    for i in range(cols):
        if (stds_spam[i] < 0.0001):
            stds_spam[i] = 0.0001
        if (stds_not_spam[i] < 0.0001):
            stds_not_spam[i] = 0.0001
    predictions = classify_data(test_data, prior_spam, prior_not_spam, means_spam, means_not_spam, stds_spam, stds_not_spam)
    confusion_matrix = calculate_confusion(predictions, test_labels)
    print("Confusion Matrix:")
    print(confusion_matrix)
    PCA(data, labels, permutation)


def compute_prior_spam(labels):
    rows = labels.shape[0]
    count = np.sum(labels)
    return count / rows
def PCA(data, labels, permutation):
    rows, cols = data.shape
    data = data[permutation]
    labels = labels[permutation]
    training_data = data[0:2300, :]
    one_vec_train = np.ones(2300)
    one_vec_test = np.ones(2301)
    test_data = data[2300:,:]
    training_labels = labels[0:2300]
    test_labels = labels[2300:]
    means_data = np.mean(training_data, axis=0)
    training_data = training_data - np.outer(one_vec_train, means_data)
    test_data = test_data - np.outer(one_vec_test, means_data) #center test data around training mean
    prior_spam = compute_prior_spam(training_labels)
    prior_not_spam= 1 - prior_spam
    covariance_matrix = np.cov(training_data, rowvar=False)
    eigenvals, eigenvectors = np.linalg.eigh(covariance_matrix)
    eigenvectors = eigenvectors[:, np.abs(eigenvals[:]) > 0.1] #PCA dropoff eigenvectors with low magnitude eigenvalues
    training_data = np.matmul(training_data, eigenvectors) #change of coordinates to eigenvector basis
    test_data = np.matmul(test_data, eigenvectors) #note that this is a change of coordinates to the eigenvector basis. 
    #The change of basis does NOT do something to the test set that we could not do one at a time for each test datum
    means_spam = np.mean(training_data[training_labels[:] == 1, :], axis=0)
    means_not_spam = np.mean(training_data[training_labels[:] == 0, :], axis=0)
    stds_spam = np.std(training_data[training_labels[:] == 1, :], axis=0)
    stds_not_spam = np.std(training_data[training_labels[:] == 0, :], axis=0)
    for i in range(training_data.shape[1]):
        if (stds_spam[i] < 0.0001):
            stds_spam[i] = 0.0001
        if (stds_not_spam[i] < 0.0001):
            stds_not_spam[i] = 0.0001
    predictions = classify_data(test_data, prior_spam, prior_not_spam, means_spam, means_not_spam, stds_spam, stds_not_spam)
    confusion_matrix = calculate_confusion(predictions, test_labels)
    print("Confusion Matrix after PCA:")
    print(confusion_matrix)
    print("PCA dimensions:")
    print(eigenvectors.shape[1])


def classify_data(data, prior_spam, prior_not_spam, means_spam, means_not_spam, stds_spam, stds_not_spam):
    (rows, cols) = data.shape
    prediction = np.zeros(rows)
    for p in range(rows): #for each data point p in test set
        class_spam = np.log(N(data[p,:], means_spam[:], stds_spam[:]))
        class_spam = np.sum(class_spam) + np.log(prior_spam)
        class_not_spam = np.log(N(data[p,:], means_not_spam[:], stds_not_spam[:]))
        class_not_spam = np.sum(class_not_spam) + np.log(prior_not_spam)
        if (class_spam > class_not_spam):
            prediction[p] = 1
    return prediction
    
def calculate_confusion(predictions, labels):
    confusion_matrix = np.zeros((2,2))
    for p in range(labels.shape[0]): #for each data point p
        confusion_matrix[int(labels[p]), int(predictions[p])] += 1
    return confusion_matrix

def N(x, mu, std):
    val = np.divide(1, np.multiply(np.sqrt(2*np.pi), std))
    val2 = np.divide(np.square(np.subtract(x, mu)), np.multiply(2, np.square(std)))
    val2 = np.exp(-val2)
    return np.multiply(val, val2)

def read_data(filename):
    return np.loadtxt(open(filename, "rb"), delimiter=",")

if __name__ == "__main__":
    main()
