import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class KNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        
    def _euclidean_metric(self, x1, x2):
        '''
        Compute the Euclidean distance
        x1, x2 -> Vector in the feature space
        '''
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def _manhattan_distance(self, x1, x2):
        '''
        Compute the Manhattan distance
        x1, x2 -> Vector in the feature space
        '''
        return np.sum(np.abs(x1 - x2))
    
    def _minkowski_distance(self, x1, x2):
        '''
        Compute the Minkowski distance
        x1, x2 -> Vector in the feature space
        '''
        return np.sum(np.abs(x1 - x2)**self.k) ** (1/self.k)
    
    def fit(self, X, y):
        '''
        Fit the model using X as training data and y as target values
        '''
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        '''
        Predict the class labels for the provided data
        '''
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    
    def _predict(self, x):
        '''
        Predict the class label for a single sample
        '''
        if self.distance_metric == 'euclidean':
            distances = [self._euclidean_metric(x, x_train) for x_train in self.X_train]
        elif self.distance_metric == 'manhattan':
            distances = [self._manhattan_distance(x, x_train) for x_train in self.X_train]
        elif self.distance_metric == 'minkowski':
            distances = [self._minkowski_distance(x, x_train) for x_train in self.X_train]
        else:
            raise ValueError("Invalid distance metric.")

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Classification
        if self.y_train.dtype == int:        
            most_common = Counter(k_nearest_labels).most_common(1)
            return most_common[0][0] 
        # Regression
        elif self.y_train.dtype == float:
            mean_labels = np.mean(k_nearest_labels)
            return mean_labels
    
class WeightedKNN(KNN):
    '''
        Assign a weight to each of the k-nearest neighbors inversely proportional to their distance from the query point. 
        In simpler terms, closer neighbors get higher weights, while farther neighbors get lower weights.
    '''
    def _predict(self, x):
        '''
        Predict the class label for a single sample
        '''
        if self.distance_metric == 'euclidean':
            distances = [self._euclidean_metric(x, x_train) for x_train in self.X_train]
        elif self.distance_metric == 'manhattan':
            distances = [self._manhattan_distance(x, x_train) for x_train in self.X_train]
        elif self.distance_metric == 'minkowski':
            distances = [self._minkowski_distance(x, x_train) for x_train in self.X_train]
        else:
            raise ValueError("Invalid distance metric.")
              
        k_distances = np.sort(distances)[:self.k]
        k_weighted = np.array(1 - k_distances/np.sum(k_distances))
        print(k_distances, k_weighted)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
             
        # Classification
        if self.y_train.dtype == int:        
            dict = {}
            for i, labels in enumerate(k_nearest_labels):
                if not labels in dict:
                    dict[labels] = k_weighted[i]
                else:
                    dict[labels] += k_weighted[i]
            most_common_weighted = [labels for labels, weights in dict.items() if weights == max(dict.values())]
            return most_common_weighted    
            
        # Regression
        elif self.y_train.dtype == float:
            mean_labels_weighted = np.mean(k_weighted*k_nearest_labels)
            return mean_labels_weighted

 
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

knn = KNN(distance_metric='manhattan')
knn.fit(X_train_std, y_train)
y_pred = knn.predict(X_test_std)
acc = accuracy_score(y_test, y_pred)

