import numpy as np

class LogisticRegression():
    def __init__(self, lr=0.01):
        self.w = None
        self.b = None
        self.lr = lr
    def sigmoid(z):
        return 1/1+np.exp(-z)
        
    def fit(self, x, y):
        samples, features = x.Shape
        self.w = np.zeros(features)
        self.b = 0
        
        for _ in range(500):
            z = np.dot(x, w) + self.b
            prediction = sigmoid(z)

            dw = (1/samples)*np.dot(x.T, prediction-y)
            db = (1/samples)*np.sum(prediction-y)

            self.w = self.w - self.lr*dw
            self.b = self.b - self.lr*db
    
    def predict(self):
        z = np.dot(x, w) + self.b
        prediction = sigmoid(z)
        cls = [0 if y<0.5 else 1 for y in prediction]
        return cls
    
        
        
        
        