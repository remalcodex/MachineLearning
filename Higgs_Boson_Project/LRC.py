import numpy as np
import random
import math
class LRC:

	def __init__(self,d,g,s):
		self.w = np.random.rand(1,d)[0]
		self.w = self.w * 0.02
		self.w = self.w - 0.01
		self.bias = random.uniform(-0.01,0.01)
		self.gamma = g
		self.sigma = s

	def predict(self,x):
		#print (self.w.dot(x)[0],self.bias)
		y=self.bias

		for i in x:
			#print(i)
			y += self.w[i]
		#y = self.w.dot(x) + self.bias
		#if math.isnan(y): print('here')
		return y

	def update(self,y_new,x):
		#print(self.w + self.eta * y_new * x)
		
		self.w = (-2/self.sigma*self.gamma+1)*self.w
		#print(y_new,self.predict(x))
		index = y_new*self.predict(x)
		if index>0:
			factor = math.exp(-1*index)/(1+math.exp(-1*index))
		else:
			factor = 1/(1+math.exp(index))
		# if index<0:
		# 	factor = -1*math.exp(index)/(1+math.exp(index))
		# else:
		# 	factor = -1/(1+math.exp(-1*index))		

		for i in x:
			self.w[i]+=self.gamma*y_new*factor
		self.bias = (-2/self.sigma*self.gamma+1)*self.bias + self.gamma*y_new*factor


	def lowupdate(self):
		#print(self.w + self.eta * y_new * x)
		self.w = self.w*(1-self.gamma) 
		self.bias = self.bias*(1-self.gamma)
