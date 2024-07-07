import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ID3Classifier:
    def __init__(self):
        self.tree = None

    class Node:
        def __init__(self, feature=None, threshold=None, value=None, left=None, right=None):
            self.feature = feature  # Index of feature to split on
            self.threshold = threshold  # Threshold value for the split
            self.value = value  # Predicted class (for leaf nodes)
            self.left = left  # Left child node
            self.right = right  # Right child node

    def __entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def __information_gain(self, X, y, feature, threshold):
        mask = X[:, feature] <= threshold
        
        left_entropy = self.__entropy(y[mask])
        right_entropy = self.__entropy(y[~mask])
        parent_entropy = self.__entropy(y)
        weight_left = sum(mask) / len(y)
        weight_right = sum(~mask) / len(y)
        return parent_entropy - (weight_left * left_entropy + weight_right * right_entropy)

    def __find_best_split(self, X, y):
        num_features = X.shape[1]
        best_feature, best_threshold, max_gain = None, None, -1

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self.__information_gain(X, y, feature, threshold)
                if gain > max_gain:
                    best_feature, best_threshold, max_gain = feature, threshold, gain

        return best_feature, best_threshold

    def __build_tree(self, X, y, depth=0, max_depth=None):
        if depth == max_depth or len(set(y)) == 1:
            return self.Node(value=np.argmax(np.bincount(y)))

        feature, threshold = self.__find_best_split(X, y)
        if feature is None:
            return self.Node(value=np.argmax(np.bincount(y)))

        mask = X[:, feature] <= threshold
        left = self.__build_tree(X[mask], y[mask], depth + 1, max_depth)
        right = self.__build_tree(X[~mask], y[~mask], depth + 1, max_depth)

        return self.Node(feature=feature, threshold=threshold, left=left, right=right)

    def fit(self, X, y, max_depth=None):
        self.tree = self.__build_tree(X, y, max_depth=max_depth)

    def __predict_one(self, node, sample):
        if node.value is not None:
            return node.value
        if sample[node.feature] <= node.threshold:
            return self.__predict_one(node.left, sample)
        else:
            return self.__predict_one(node.right, sample)

    def predict(self, X):
        return [self.__predict_one(self.tree, sample) for sample in X]
        
if __name__ == '__main__':
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.2)

    clf = ID3Classifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.2%}')