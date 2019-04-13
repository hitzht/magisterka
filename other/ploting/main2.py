import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple


n_groups = 5

iter500 = (32.72, 50.72, 63.33, 56.77, 65.92)
iter1000 = (24.65, 46.09, 62.46, 56.04, 65.87)
iter5000 = (18.93, 39.47, 60.07, 52.56, 63.53)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.2

rects1 = ax.bar(index, iter500, bar_width, color='g', label='500 iteracji')
rects2 = ax.bar(index + bar_width, iter1000, bar_width, color='b', label='1000 iteracji')
rects3 = ax.bar(index + 2 * bar_width, iter5000, bar_width, color='r', label='5000 iteracji')


ax.set_xlabel('Instancja testowa')
ax.set_ylabel('Średni błąd[%]')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(('chr12a', 'chr15a', 'chr18a', 'chr20a', 'chr25a'))
ax.legend()

fig.tight_layout()
plt.show()