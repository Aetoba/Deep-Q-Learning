import matplotlib.pyplot as plt
import pickle

from prf_get import fromLine
from prf_get import prf_get

rewlst, epslst, count = fromLine()

print(len(rewlst))
done = False
avgs = []
maxs = []
spacing = 100
plt_lst = rewlst
while not done:
    total = 0
    maxi = 0
    n = 0
    for i in range(spacing):
        try:
            num = plt_lst.pop(0)
            total += num
            maxi = max(maxi, num)
            n += 1
        except:
            done = True
    if n > 0:
        avgs.append(total/n)
        maxs.append(maxi)


plt.plot(avgs)
plt.plot(maxs)
# plt.axis([0,50,130,290])
# plt.xticks([0,5,10,15,20])
plt.show()