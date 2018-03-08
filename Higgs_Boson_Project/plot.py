import matplotlib as plt
import numpy as np
y=[70.269,70.337,70.142,70.874,70.472]
x_coordinate = np.arange(5)
x_coordinate = [ 1 * i+1 for i in range(5) ]
plt.plot(x_coordinate,y)
plt.xticks(x_coordinate)
plt.xlabel('Train Set ID')
plt.ylabel('Test set Accuracy')
plt.show()

