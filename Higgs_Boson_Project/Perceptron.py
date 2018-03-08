import numpy as np
import random
class Perceptron:

	def __init__(self,d,e):
		self.w = np.random.rand(1,d)[0]
		self.w = self.w * 0.02
		self.w = self.w - 0.01
		self.bias = random.uniform(-0.01,0.01)
		self.eta = e

	def predict(self,x):
		#print (self.w.dot(x)[0],self.bias)
		y = self.w.dot(x) + self.bias
		return y

	def update(self,y_new,x):
		#print(self.w + self.eta * y_new * x)
		
		self.w = self.w + self.eta * y_new * x
		self.bias = self.bias + self.eta * y_new
