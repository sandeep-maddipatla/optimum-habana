import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt

data_eager = pkl.load(open("verify_eager_mnist.pkl", 'rb'))
data_lazy = pkl.load(open("verify_lazy_mnist.pkl", 'rb'))
data_cpu = pkl.load(open("verify_cpu_mnist.pkl", 'rb'))

loss_eager = data_eager[0]
loss_lazy = data_lazy[0]
loss_cpu = data_cpu[0]

plt.plot(loss_eager[0::20])
plt.plot(loss_lazy[0::20])
plt.plot(loss_cpu[0::20])
plt.legend(['hpu:eager', 'hpu:lazy', 'cpu'])

plt.yscale('log')
plt.title('MNIST Log loss')
plt.savefig('MNIST_hpu_cpu_loss_curve.png')