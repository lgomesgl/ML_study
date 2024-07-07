import numpy as np
from numpy.linalg import norm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class SVM:
    def __init__(self, kernel='linear', C = None):
        self.kernel = kernel
        self.C = C
        
    def __distance_hyperplane_point(self, point, weights_hyperplane, bias_hyperplane):
        '''
        Return the distance between the point and the hyperplane
        '''
        distance = np.abs((np.sum(np.transpose(weights_hyperplane)*point) + bias_hyperplane))/norm(weights_hyperplane)
        return distance
    
    def __distance_between_points(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2)**2))
    
    def find_the_minimum_distance(self, X, y):
        labels = np.unique(y)
        
    
    # def __find_the_best_distance(self, X):
        
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        