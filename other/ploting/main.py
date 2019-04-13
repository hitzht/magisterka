import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple


n_groups = 5

gpu_at = (0.23, 0.14, 0.14, 0.14, 0.14)
gpu_att = (3.51, 4.56, 5.93, 6.45, 9.22)
cpu_at = (12.48, 15.64, 20.20, 21.81, 27.73)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.2

rects1 = ax.bar(index, gpu_at, bar_width, color='g', label='gpu AT')
rects2 = ax.bar(index + bar_width, gpu_att, bar_width, color='b', label='gpu ATT')
rects3 = ax.bar(index + 2 * bar_width, cpu_at, bar_width, color='r', label='cpu AT')


ax.set_xlabel('Instancja testowa')
ax.set_ylabel('Czas wykonania[s]')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(('chr12a', 'chr15a', 'chr18a', 'chr20a', 'chr25a'))
ax.legend()

fig.tight_layout()
plt.show()