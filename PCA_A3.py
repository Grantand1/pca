#For HW3, we are going to visualize MNIST data using Principle Component Analysis.
# HW3 aims to visualize the MNIST data. MNIST data is high-dimensional data.
# The purpose is reduce the dimensionality of the data by using PCA
# Then plot the new projected data in the plot

#Andrew Grant
#Assignment 3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


###Read in the data with
dataset = pd.read_csv('MNIST_100.csv')
#print (dataset)


###spliting the data
data = dataset.iloc[1:1000, 0:785]
label = dataset.iloc[:,0]
#print (data)
#print(label)

###Feature Scaling

s_scaler = StandardScaler()
data_x = s_scaler.fit_transform(data)
#print(data_x)

###calculate covariance matrix
#the covariance matrix functon can be calculated using the numpy library
#and transposing the feature scaling data
cov_matrix = np.cov(data_x.T)

###calculate the eigen vector and eigen value
eigen_value, eigen_vector = np.linalg.eig(cov_matrix)
#print(eigen_value)
#print(eigen_vector)

###sort eigen value/vector in descending order
#for sorting find the absolute value of each varable in the matrix
abs_eigen_value = np.absolute(eigen_value)
abs_eigen_vector = np.absolute(eigen_vector)
sorted_abs_eigen_value = sorted(abs_eigen_value, reverse=True)
#print('sorted', sorted_abs_eigen_value)
sorted_abs_eigen_vector = np.sort(abs_eigen_vector)[::-1]
#print(sorted_abs_eigen_vector)

#test
#print(np.sort(abs_eigen_vector)[0] == np.sort(abs_eigen_vector)[::-1][0])
#Get the highest values in the eigenvector matrix and eigenvalue matrix
first_pair = [sorted_abs_eigen_value[0], sorted_abs_eigen_vector[:,0]]
second_pair = [sorted_abs_eigen_value[1], sorted_abs_eigen_vector[:,1]]
#print(first_pair, second_pair)

### eigenvector multiply by the original dataset
pca_1 = data.dot(eigen_vector[:,0])
pca_2 = data.dot(eigen_vector[:,1])

#test
#new_eigen_1 = sorted_abs_eigen_vector[:,0].dot(sorted_abs_eigen_value[0])
#new_eigen_2 = sorted_abs_eigen_vector[:,1].dot(sorted_abs_eigen_value[1])
#new_pca_1 = data.dot(new_eigen_1)
#new_pca_2 = data.dot(new_eigen_2)


### Plot
color= ["magenta", "red", "green", "black", "teal","orange","purple","pink","brown","blue"]
color_array = []
for i in range(len(label)):
    color_array.append(color[label[i] - 1])

plt.scatter(pca_1, pca_2, c=color_array)
plt.show()











