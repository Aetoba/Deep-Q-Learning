import matplotlib.pyplot as plt
import pickle

from prg_get import fromLine
from prg_get import prg_get

t_stplst, epilst, stplst, rewlst, count = fromLine()

done = False
avgs = []
spacing = 100
plt_lst = rewlst
while not done:
    total = 0
    n = 0
    for i in range(spacing):
        try:
            total += plt_lst.pop(0)
            n += 1
        except:
            done = True
    if n > 0:
        avgs.append(total/n)

plt.plot(avgs)
# plt.axis([0,50,100,180])
# plt.xticks([0,5,10,15,20])
plt.show()
