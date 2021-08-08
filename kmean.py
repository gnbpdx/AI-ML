import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
def plot(data, cluster_value, k):
    x_val = data[:,0]
    y_val = data[:,1]
    colors = cm.rainbow(np.linspace(0,1,k))
    for i in range(k):
        x = x_val[cluster_value[:] == i]
        y = y_val[cluster_value[:] == i]
        color = colors[i,:]
        plt.scatter(x,y, c=color)
    plt.show()

def read_in(filename):
    data = np.genfromtxt(filename, delimiter=None)
    return data
def kmeans(data, k):
    cluster_value = np.random.randint(0,k, data.shape[0])
    cluster_mean = np.zeros((k,2))
    repeat = True
    previous_loss = 2e10
    while (repeat == True):  
        for i in range(10):
            for n in range(k):
                cluster_mean[n,:] = np.mean(data[cluster_value[:] == n], axis=0)
            data_distance = np.zeros((data.shape[0], k))
            loss = 0
            for i in range(data.shape[0]):
                for j in range(k):
                    data_distance[i,j] = np.linalg.norm(data[i] - cluster_mean[j])
            for i in range(data.shape[0]):
                loss += data_distance[i,cluster_value[i]]
            if (previous_loss - loss < 1): #algorithm has converged
                repeat = False
            previous_loss = loss
            for i in range(data.shape[0]):
                cluster_value[i] = np.argmin(data_distance[i,:])
    return cluster_value, previous_loss
def cmeans(data, c, m): #fuzzy c-means
    one = np.ones(c)
    cluster = np.zeros(data.shape[0])
    cluster_value = np.zeros((data.shape[0], c))
    previous_loss = 2e10
    for i in range(data.shape[0]):
        cluster_value[i,:] = np.random.dirichlet(one)
    repeat = True
    while(repeat == True): #repeat until convergence is met
        centroid = np.zeros((c,2))
        for i in range(c):
            for j in range(data.shape[0]):
                centroid[i,:] += np.multiply(np.power(cluster_value[j,i], m), data[j,:])
            sum = 0
            for j in range(data.shape[0]):
                sum += np.power(cluster_value[j,i], m)
            centroid[i,:] = np.divide(centroid[i,:], sum) 
        for i in range(data.shape[0]):
            for j in range(c):
                sum = 0
                temp = np.abs(np.linalg.norm(data[i,:] - centroid[j,:]))
                for k in range(c):
                    temp2 = np.abs(np.linalg.norm(data[i,:] - centroid[k,:]))
                    val = temp / temp2
                    val = np.power(val, (2/(m-1)))
                    sum += val
                cluster_value[i,j] = 1 / sum
        for i in range(data.shape[0]):
            cluster[i] = np.argmax(cluster_value[i,:])
        loss = 0
        for i in range(data.shape[0]):
            loss += np.linalg.norm(data[i,:] - centroid[int(cluster[i])]) #calculate loss 
        if (previous_loss - loss < 1): 
            repeat = False
        previous_loss = loss
    return cluster, loss
def main():
    data = read_in('data/cluster_dataset.txt')
    k = 3
    loss = np.NaN
    cluster_value = None
    for i in range(10):
        new_cluster_value, new_loss = cmeans(data, k, 3)
        if (np.isnan(loss)):
            loss = new_loss
            cluster_value = new_cluster_value
        if (new_loss < loss):
            loss = new_loss
            cluster_value = new_cluster_value
    print("c-means loss: ")
    print(loss)
    plot(data, cluster_value, k)
    loss = np.NaN
    for i in range(10):
        new_cluster_value, new_loss = kmeans(data, k)
        if (np.isnan(loss)):
            loss = new_loss
            cluster_value = new_cluster_value
        if (new_loss < loss):
            loss = new_loss
            cluster_value = new_cluster_value
    print("k-means loss: ")
    print(loss)
    plot(data, cluster_value, k)

if __name__ == '__main__':
    main()
