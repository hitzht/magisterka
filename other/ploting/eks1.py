import numpy as np
import matplotlib.pyplot as plt

# plt.style.use('seaborn-whitegrid')

iterations = [1, 2, 3]

chr22a_time = [41, 458, 2927]
esc32a_time = [110, 765, 5289]
esc64a_time = [64, 2461, 18687]
ste36a_time = [72, 929, 6674]
lipa40b_time = [86, 1088, 7725]

fig = plt.figure(figsize=(10, 6))
ax = plt.axes()


plt.plot(iterations, chr22a_time, color="green", linestyle='solid')
plt.plot(iterations, esc32a_time, color="red", linestyle='solid')
plt.plot(iterations, esc64a_time, color="orange", linestyle='solid')
plt.plot(iterations, ste36a_time, color="blue", linestyle='solid')
plt.plot(iterations, lipa40b_time, color="brown", linestyle='solid')

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

ax.legend(["chr22a",
           "esc32a",
           "esc64a",
           "ste36a",
           "lipa40b",
           ], loc='center left', bbox_to_anchor=(1, 0.5))


plt.xlabel("Numer eksperymentu")
plt.ylabel("Czas obliczeń [ms]")
plt.axis((1, 3, 0, 20000))
plt.grid()
plt.xticks([1, 2, 3])
plt.show()