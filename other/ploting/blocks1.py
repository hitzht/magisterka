import numpy as np
import matplotlib.pyplot as plt

# plt.style.use('seaborn-whitegrid')

blocks = [10, 25, 50, 75, 100, 150, 200, 250, 300]
chr22a_error = [12, 12, 11, 9, 10, 8.1, 9.6, 9.1, 8]
chr22a_avg_error = [15, 14, 13, 12, 12, 12, 11, 11, 10]

esc32a_error = [42, 37, 28, 32, 35, 32, 31, 34, 26]
esc32a_avg_error = [48, 42, 40, 38, 41, 38, 37, 37, 34]

ste36a_error = [28, 29, 27, 26, 24, 27, 27, 24, 24]
ste36a_avg_error = [35, 32, 31, 31, 30, 29, 30, 29, 29]

wil50_error = [5.5, 4.9, 4.9, 5.2, 4.4, 4.3, 4.8, 3.9, 4.8]
wil50_avg_error = [5.9, 5.5, 5.4, 5.5, 5.2, 5.2, 5.2, 5.2, 5.1]


fig = plt.figure(figsize=(10, 6))
ax = plt.axes()


plt.plot(blocks, esc32a_error, color="green", linestyle='solid')
plt.plot(blocks, esc32a_avg_error, color='green', linestyle='dashed')

plt.plot(blocks, ste36a_error, color="red", linestyle='solid')
plt.plot(blocks, ste36a_avg_error, color="red", linestyle='dashed')

plt.plot(blocks, chr22a_error, color='blue', linestyle='solid')
plt.plot(blocks, chr22a_avg_error, color='blue', linestyle='dashed')

plt.plot(blocks, wil50_error, color="orange", linestyle='solid')
plt.plot(blocks, wil50_avg_error, color="orange", linestyle='dashed')

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

# Put a legend to the right of the current axis
# ax.legend()

ax.legend(["esc32a",
           "esc32a",
           "ste36a",
           "ste36a",
           "chr22a",
           "chr22a",
           "wil50",
           "wil50"], loc='center left', bbox_to_anchor=(1, 0.5))


plt.axis((10, 300, 0, 50))
plt.xlabel("Liczba bloków")
plt.ylabel("Błąd względny [%]")
plt.grid()
plt.show()