import matplotlib.pyplot as plt
import pickle

from mta_get import fromLine
from mta_get import mta_get

loss_avgs, loss_maxs, val_avgs, val_maxs, count = fromLine()

done = False
avgs = []
maxs = []
spacing = 100
plt_list = loss_avgs
while not done:
    total = 0
    maxi = 0
    n = 0
    for i in range(spacing):
        try:
            val = plt_list.pop(0)
            total += val
            maxi = max(val, maxi)
            n += 1
        except:
            done = True
    if n > 0:
        avgs.append(total/n)
        maxs.append(maxi)

plt.plot(avgs)
plt.plot(maxs)
# plt.axis([0,50,0,0.26])
# plt.yticks([i/100 for i in range(0, 28, 2)])
plt.show()
