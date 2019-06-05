import matplotlib.pyplot as plt
import numpy as np
data=np.load('tmp.npy')
print(data.shape)
data=data[:,:,0]
plt.imshow(data)
plt.show()