import numpy as np
import pandas as pd
import statsmodels.api as sm

class MCP(object):  
    """
    COORDINATE DESCENT ALGORITHMS FOR NONCONVEX PENALIZED REGRESSION, 
    WITH APPLICATIONS TO BIOLOGICAL FEATURE SELECTION
    """
    def __init__(self, x, y, lambd, gamma = 2, iteration = 100):
        self.iteration = iteration
        assert ~any(np.isnan(x).ravel())
        assert ~any(np.isnan(y))
        assert lambd > 0
        self.x = x
        self.y = y
        assert self.y.shape[0] == self.x.shape[0]
        self.n = self.y.shape[0]
        self.p = self.x.shape[1]
        self.lambd = lambd
        self.gamma = float(gamma)
        self.__standard()
        
    def __standard(self):
        self.ymean = self.y.mean()
        self.reg_y = self.y - self.ymean
        self.xmean = self.x.mean(axis = 0)
        self.xvar = self.x.var(axis = 0)
        self.reg_x = (self.x - self.xmean)/(np.sqrt(self.xvar))
    
    def __S(self, z):
        if z > self.lambd:
            return z - self.lambd
        elif z < ((-1) * self.lambd):
            return z + self.lambd
        else:
            return 0        
    
    def __fmcp(self, z):
        if np.abs(z) > (self.gamma * self.lambd):
            return z
        else:
            return (self.__S(z) / (1 - 1.0 / self.gamma))
    
    def fit(self):
        self.initial_model = sm.WLS(self.reg_y, self.reg_x).fit()
        self.params = self.initial_model.params
        self.resid = self.initial_model.resid
        for m in xrange(self.iteration):
            self.Z = np.dot(self.reg_x.T, self.resid) / self.n + self.params
            self.params = np.array(map(self.__fmcp, self.Z))
            self.resid = self.reg_y - np.dot(self.reg_x, self.params)
        self._intercept = self.ymean - np.sum(self.params * self.xmean / np.sqrt(self.xvar))
        self._coef = self.params / np.sqrt(self.xvar)
    
    def predict(self, x_pre):
        return np.dot(x_pre, self._coef) + self._intercept

class SCAD(object):  
    """
    COORDINATE DESCENT ALGORITHMS FOR NONCONVEX PENALIZED REGRESSION, 
    WITH APPLICATIONS TO BIOLOGICAL FEATURE SELECTION
    """
    def __init__(self, x, y, lambd, gamma = 3.7, iteration = 100):
        self.iteration = iteration
        assert ~any(np.isnan(x).ravel())
        assert ~any(np.isnan(y))
        assert lambd > 0
        self.x = x
        self.y = y
        assert self.y.shape[0] == self.x.shape[0]
        self.n = self.y.shape[0]
        self.p = self.x.shape[1]
        self.lambd = lambd
        self.gamma = float(gamma)
        self.__standard()
        
    def __standard(self):
        self.ymean = self.y.mean()
        self.reg_y = self.y - self.ymean
        self.xmean = self.x.mean(axis = 0)
        self.xvar = self.x.var(axis = 0)
        self.reg_x = (self.x - self.xmean)/(np.sqrt(self.xvar))
    
    def __S(self, z, lambd):
        if z > lambd:
            return z - lambd
        elif z < ((-1) * lambd):
            return z + lambd
        else:
            return 0        
    
    def __fscad(self, z):
        if np.abs(z) > (self.gamma * self.lambd):
            return z
        elif np.abs(z) <= (2 * self.lambd):
            return self.__S(z, self.lambd)
        else:
            return self.__S(z, (self.gamma * self.lambd / (self.gamma - 1)) / (1 - 1 / (self.gamma - 1)))
    
    def fit(self):
        self.initial_model = sm.WLS(self.reg_y, self.reg_x).fit()
        self.params = self.initial_model.params
        self.resid = self.initial_model.resid
        for m in xrange(self.iteration):
            self.Z = np.dot(self.reg_x.T, self.resid) / self.n + self.params
            self.params = np.array(map(self.__fscad, self.Z))
            self.resid = self.reg_y - np.dot(self.reg_x, self.params)
        self._intercept = self.ymean - np.sum(self.params * self.xmean / np.sqrt(self.xvar))
        self._coef = self.params / np.sqrt(self.xvar)
    
    def predict(self, x_pre):
        return np.dot(x_pre, self._coef) + self._intercept