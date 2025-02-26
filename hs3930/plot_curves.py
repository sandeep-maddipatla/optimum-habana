import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt
import os

suffix = os.getenv('SUFFIX','')
data_eager = pkl.load(open(f"verify_eager_{suffix}.pkl", 'rb'))
data_lazy = pkl.load(open(f"verify_lazy_{suffix}.pkl", 'rb'))
data_cpu = pkl.load(open(f"verify_cpu_{suffix}.pkl", 'rb'))

loss_eager = data_eager[0]
loss_lazy = data_lazy[0]
loss_cpu = data_cpu[0]

plt.plot(loss_eager[:18000])
plt.plot(loss_lazy[:18000])
plt.plot(loss_cpu[:18000])
plt.legend(['hpu:eager', 'hpu:lazy', 'cpu'])


plt.yscale('log')
e = suffix.replace('_',' ')
plt.title(f'{e} Log loss')
plt.savefig(f'{suffix}_loss_curve.png')