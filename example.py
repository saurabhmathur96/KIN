import numpy as np
from noisy_or import NoisyOrClassifier, MonotonicNoisyOrClassifier
from sklearn.metrics import roc_auc_score
from scipy.special import expit as sigmoid

np.random.seed(0)
N = 100
n = 4
X_train = np.random.randint(0, 4, (N, n))
y_train = np.random.randint(0, 2, N)

X_test = np.random.randint(0, 4, (N, n))
y_test = (sigmoid(X_test[:, 0] + X_test[:, 1] - X_test[:, 2]) > 0.5).astype(int)

nor = NoisyOrClassifier()
nor.fit(X_train, y_train)
score = roc_auc_score(y_test, nor.predict_proba(X_test)[:, 1])
print (f"Noisy-Or AUC-ROC = {score:6.4f}")

constraints = np.array([+1, +1, -1, 0])
lambda_, epsilon = 10, 0.001
nor = MonotonicNoisyOrClassifier(constraints, lambda_, epsilon)
nor.fit(X_train, y_train)

for i in range(1, 10):
  print(f"Penalty = {nor.penalty:6.4e}")
  if np.isclose(nor.penalty, 0):
    break
  nor = MonotonicNoisyOrClassifier(constraints, lambda_*(10**i), epsilon)
  nor.fit(X_train, y_train)
score = roc_auc_score(y_test, nor.predict_proba(X_test)[:, 1])
print (f"Knowledge Intensive Noisy-Or AUC-ROC = {score:6.4f}")