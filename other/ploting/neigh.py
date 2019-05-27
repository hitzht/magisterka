import numpy as np
import matplotlib.pyplot as plt

# plt.style.use('seaborn-whitegrid')

neigh_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
chr22a_error = [4.9, 7.4, 6.8, 5.9, 6.9, 6.5, 5.6, 7.7, 11, 11]
chr22a_avg_error = [8.4, 9.1, 8.6, 8.9, 8.5, 7.5, 7.8, 11, 14, 14]

esc32a_error = [20, 20, 20, 20, 22, 18, 20, 23, 28, 31]
esc32a_avg_error = [25, 24, 25, 25, 25, 25, 24, 25, 37, 40]

ste36a_error = [17, 19, 17, 17, 18, 18, 16, 18, 18, 23]
ste36a_avg_error = [20, 20, 20, 20, 20, 20, 19, 20, 22, 27]

wil50_error = [3.6, 3.3, 3.3, 3.4, 3.6, 3.2, 3.5, 3.5, 3.2, 4.1]
wil50_avg_error = [3.9, 3.8, 3.9, 3.9, 4.0, 3.9, 3.9, 3.8, 4, 4.6]


fig = plt.figure(figsize=(10, 6))
ax = plt.axes()


plt.plot(neigh_size, esc32a_error, color="green", linestyle='solid')
plt.plot(neigh_size, esc32a_avg_error, color='green', linestyle='dashed')

plt.plot(neigh_size, ste36a_error, color="red", linestyle='solid')
plt.plot(neigh_size, ste36a_avg_error, color="red", linestyle='dashed')

plt.plot(neigh_size, chr22a_error, color='blue', linestyle='solid')
plt.plot(neigh_size, chr22a_avg_error, color='blue', linestyle='dashed')

plt.plot(neigh_size, wil50_error, color="orange", linestyle='solid')
plt.plot(neigh_size, wil50_avg_error, color="orange", linestyle='dashed')

# Shrink current axis by 20%
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

# Put a legend to the right of the current axis
# ax.legend()

ax.legend(["esc32a",
           "esc32a",
           "ste36a",
           "ste36a",
           "chr22a",
           "chr22a",
           "wil50",
           "wil50"])



plt.xlabel("Rozmiar otoczenia")
plt.ylabel("Błąd względny [%]")
plt.grid()
plt.axis((0.1, 1.0, 0, 45))
plt.show()