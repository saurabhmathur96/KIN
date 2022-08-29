import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from noisyor import NoisyOr, MonotonicNoisyOr

class NoisyOrClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self._model = None

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        assert self.classes_ == [0, 1]
        
        self.X_ = X
        self.y_ = y

        n = X.shape[1]

        w = np.random.uniform(0, 1, size=(n,)) # emission
        b = np.random.uniform(0, 1, size=(n,)) # emission
        q = np.random.uniform(0, 1, size=(n,)) # inhibition
        ql = np.random.uniform(0, 1) # leak
        self._model = NoisyOr(w, b, q, ql)
        self._model.fit(X, y)
        
        # Return the classifier
        return self
    
    def predict_proba(self, X):
        
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        
        p1 = self._model.predict_proba(X)

        p = np.zeros((len(p), 2))
        p[:, 1] = p1 
        p[:, 0] = 1-p1
        return p 
        

    def predict(self, X):

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        
        p = self.predict_proba(X)
        return np.array(p[:, 1] > 0.5, dtype = int)

class MonotonicNoisyOrClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, constraints, lambda_, epsilon):
        self.constraints = constraints
        self.lambda_ = lambda_ 
        self.epsilon = epsilon
        self._model = None

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        assert self.classes_ == [0, 1]
        
        self.X_ = X
        self.y_ = y

        n = X.shape[1]

        w = np.random.uniform(0, 1, size=(n,)) # emission
        b = np.random.uniform(0, 1, size=(n,)) # emission
        q = np.random.uniform(0, 1, size=(n,)) # inhibition
        ql = np.random.uniform(0, 1) # leak
        self._model = MonotonicNoisyOr(w, b, q, ql, self.constraints, self.lambda_, self.epsilon)
        self._model.fit(X, y)
        
        # Return the classifier
        return self
    
    def predict_proba(self, X):
        
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        
        p1 = self._model.predict_proba(X)

        p = np.zeros((len(p), 2))
        p[:, 1] = p1 
        p[:, 0] = 1-p1
        return p 
        

    def predict(self, X):

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        
        p = self.predict_proba(X)
        return np.array(p[:, 1] > 0.5, dtype = int)
    

    @property
    def penalty(self):
        return self._model.penalty