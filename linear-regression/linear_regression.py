import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, iters=1):
        self.lr = lr
        self.iters = iters
        self.w = None
        self.b = None
    def fit(self, x, y):
        print(x)
        x_sample, x_features = x.shape
        self.w = np.zeros(x_features)
        print(self.w)
        self.b = 0
        
        while self.iters:
            y_pred = np.dot(x, self.w) + self.b
            print(np.dot(x,self.w))
            print(y_pred)
            dw = (1/x_sample)*np.dot(x.T, (y-y_pred))
            db = (1/x_sample)*np.sum(y-y_pred)
        
            self.w = self.w - self.lr*dw
            self.b = self.b - self.lr*db
    def predict(self):
        y_pred = np.dot(x, self.w) + self.b
        return y_pred
