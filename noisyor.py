import numpy as np
from scipy.special import xlogy, expit 
from tqdm import tqdm, trange
from scipy.optimize import minimize

from itertools import product 


def sgm(x): return expit(x)
def sgm_grad(x): return sgm(x)*(1-sgm(x))



class NoisyOr:
  def __init__(self, w, b, q, ql):
    self.w = w
    self.b = b
    self.q = q
    self.ql = ql
  
  def unpack(self, x):
    n = len(self.q)
    w, b, q, ql = np.split(x, [n, 2*n, 3*n])
    
    self.w, self.b = w, b
    self.q, self.ql = q, ql[0]

  @property
  def params(self):
    return np.concatenate([self.w, self.b, self.q, np.array([self.ql]) ])
  
  def P(self, X):
    n = len(self.q)
    p0 = np.ones(len(X))*(1-sgm(self.ql))
    
    q = sgm(self.q)
    for j in range(n):
      px = sgm(X[:, j]*self.w[j] + self.b[j])
      p0 *= px*q[j] + (1-px)
    p1 = 1 - p0
    return np.array([p0, p1]).T
  
  def loglik(self, X, y):
    p = self.P(X)
    p0, p1 = p[:, 0], p[:, 1]
    return np.sum(xlogy(y, p1) + xlogy(1-y, p0), axis=0)

  def grad(self, X, y):
    N = len(y)
    n = len(self.q)
    grad = 0
    pp = self.P(X)
    p0, p1 = pp[:, 0], pp[:, 1]

    ql_grad = sgm_grad(self.ql)
    ql = sgm(self.ql)

    E = 0

    first = -y/(p1+E) + (1-y)/(p0+E)
    second = p0/(1-ql+E)
    third = -1
    grad_ql = np.sum(first*second*third*ql_grad, axis=0)
    

    q = sgm(self.q)
    q_grad = sgm_grad(self.q)
    grad_q = np.zeros(n)
    for j in range(n):
      px = sgm(X[:, j]*self.w[j] + self.b[j])
      second = p0/(px*q[j] + 1 - px + E)
      third = px
      grad_q[j] = np.sum(first*second*third*q_grad[j], axis=0)
    
    
    grad_w = np.zeros(n)
    grad_b = np.zeros(n)

    for j in range(n):
      ax = X[:, j]*self.w[j] + self.b[j]
      px = sgm(ax)
      px_grad = sgm_grad(ax)
      second = p0/(px*q[j] + 1 - px + E)
      third = (q[j] - 1)
      c = first*second*third*px_grad
      grad_w[j] = np.sum(c*X[:, j], axis=0)
      grad_b[j] = np.sum(c, axis=0)
    
    return np.concatenate([grad_w, grad_b, grad_q, np.array([grad_ql])])
  
  def fit(self, X, y):

    def g(x):
      self.unpack(x)
      return -self.grad(X, y)

    def f(x):
      self.unpack(x)
      return -self.loglik(X, y)
    
    res = minimize(f, self.params, jac=g, method='L-BFGS-B')
    self.unpack(res.x)

class MonotonicNoisyOr(NoisyOr):
  def __init__(self, w, b, q, ql, constraints, lamda = 1, eps = 0.01):
    super().__init__(w, b, q, ql)
    self.lamda = lamda
    self.eps = eps
    self.constraints = constraints

  def delta(self, i, a, b):
    first = 1-sgm(self.w[i]*a + self.b[i])
    second = 1-sgm(self.w[i]*b + self.b[i])
    if self.constraints[i] == 1:
      return first - second + self.eps
    elif self.constraints[i] == -1:
      return -first + second + self.eps
    else:
      return 0
  
  def delta_grad(self, i, a, b):
    first = -sgm_grad(self.w[i]*a + self.b[i])
    second = -sgm_grad(self.w[i]*b + self.b[i])
    if self.constraints[i] == 1:
      return first*a - second*b, first - second
    elif self.constraints[i] == -1:
      return -first*a + second*b, -first + second
    else:
      return 0, 0

  def penalty(self, X):
    P = 0
    n = len(self.q)
    for j in range(n):
      v = np.unique(X[:, j])
      for a, b in ( (a, b) for a, b in product(v, v) if a > b):
        delta = self.delta(j, a, b)
        condition = (delta > 0)
        P += condition*delta**2 # |delta|
    return P
  
  def grad(self, X, y):
    # P = \sum condition 2*log (delta_i)
    # grad P = \sum condition (2/delta_i) grad delta_i
    grad = super().grad(X, y)
    n = len(self.q)
    for j in range(n):
      if self.constraints[j] == 0:
        continue
      
      
      v = np.unique(X[:, j])
      for a, b in ( (a, b) for a, b in product(v, v) if a > b):
        if self.constraints[j]==0: continue
        
        dw, db = self.delta_grad(j, a, b)
        delta = self.delta(j, a, b)
        
        if delta <= 0: continue

        grad[j]   -= self.lamda*dw*2*delta
        grad[n+j] -= self.lamda*db*2*delta
    return grad
  
  def fit(self, X, y):

    def g(x):
      self.unpack(x)
      # print (-self.grad(X, y))
      return -self.grad(X, y)

    def f(x):
      self.unpack(x)
      
      return -self.loglik(X, y) + self.lamda*self.penalty(X)
    
    res = minimize(f, self.params, jac=g, method='L-BFGS-B')
    self.unpack(res.x)
