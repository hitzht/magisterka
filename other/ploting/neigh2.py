import numpy as np
import matplotlib.pyplot as plt

# plt.style.use('seaborn-whitegrid')

neigh_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

chr22a_time = [1400, 1405, 1402, 1402, 1386, 1387, 1405, 1583, 4493, 26032]

esc32a_time = [2588, 2592, 2592, 2645, 2574, 2574, 2575, 2605, 5033, 52501]

ste36a_time = [3252, 3250, 3253, 3254, 3276, 3228, 3231, 3247, 9208, 65569]

wil50_error = [5528, 5529, 5636, 5531, 5540, 5532, 5532, 5531, 8677, 142838]

fig = plt.figure(figsize=(10, 6))
ax = plt.axes()

plt.plot(neigh_size, esc32a_time, color="green", linestyle='solid')
plt.plot(neigh_size, ste36a_time, color="red", linestyle='solid')
plt.plot(neigh_size, chr22a_time, color='blue', linestyle='solid')
plt.plot(neigh_size, wil50_error, color="orange", linestyle='solid')

ax.legend(["esc32a", "ste36a", "chr22a", "wil50"])

plt.xlabel("Rozmiar otoczenia")
plt.ylabel("Czas obliczeń [ms]")
plt.grid()
plt.axis((0.1, 1.0, 0, 150000))

plt.show()