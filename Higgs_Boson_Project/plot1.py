# import matplotlib.pyplot as plt
# import numpy as np
# acc=[64.182,65.06,64.96,64.04,64.54]
# recall=[72.78,72.94,72.43,72.86,72.35]
# prec=[64.18,64.45,64.34,64.76,64.78]
# x_coordinate = np.arange(5)
# x_coordinate = [ 1 * i+1 for i in range(5) ]
# plt.plot(x_coordinate,acc)
# plt.plot(x_coordinate,recall)
# plt.plot(x_coordinate,prec)
# plt.xticks(x_coordinate)
# plt.xlabel('Train Set ID')
# plt.ylabel('Percentage (%)')
# plt.legend(['Accuracy', 'Recall', 'Precision'], loc='upper left')
# plt.title('Averaged Perceptron - Accuracy, Recall, Precision')
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
# acc=[72.189,72.307,72.337,72.349,72.501]
# recall=[74.918,75.0878,75.1179,75.120,75.3048]
# prec=[73.375,73.4574,73.4842,73.5501,73.621]
acc=[72.87,72.8371,74.465, 74.47, 74.81]
recall=[70.76,72.315,69.644, 72.19,71.29]
prec=[71.33,70.639,74.4425, 73.18,74.16]
x_coordinate = np.arange(5)
#x_coordinate = [ 1 * i+1 for i in range(5) ]
x_coordinate = [ 3,5,6,8,9]
plt.plot(x_coordinate,acc)
plt.plot(x_coordinate,recall)
plt.plot(x_coordinate,prec)
plt.xticks(x_coordinate)
plt.xlabel('Number of layers')
plt.ylabel('Percentage (%)')
plt.legend(['Accuracy', 'Recall', 'Precision'], loc='upper left')
plt.title('Neural Netowork with increasing number of layers.')
plt.show()

