import numpy as np
import matplotlib.pyplot as plt

# plt.style.use('seaborn-whitegrid')

iterations = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1024]
chr22a_error = [19, 15, 16, 15, 14, 15, 13, 16, 17, 14]
chr22a_avg_error = [22, 19, 18, 18, 18, 17, 17, 18, 18, 17]

esc32a_error = [54, 55, 51, 48, 51, 55, 49, 49, 54, 52]
esc32a_avg_error = [64, 63, 61, 60, 58, 61, 57, 57, 58, 57]

ste36a_error = [45, 39, 35, 33, 39, 40, 38, 31, 38, 36]
ste36a_avg_error = [47, 44, 43, 43, 43, 43, 43, 41, 41, 41]

wil50_error = [6.9, 6.1, 6.3, 5.8, 5.9, 6.2, 6.3, 6.3, 6.4, 6.2]
wil50_avg_error = [7.2, 7, 6.8, 6.8, 6.7, 6.8, 6.7, 6.8, 2.8, 6.6]


fig = plt.figure(figsize=(10, 6))
ax = plt.axes()


plt.plot(iterations, esc32a_error, color="green", linestyle='solid')
plt.plot(iterations, esc32a_avg_error, color='green', linestyle='dashed')

plt.plot(iterations, ste36a_error, color="red", linestyle='solid')
plt.plot(iterations, ste36a_avg_error, color="red", linestyle='dashed')

plt.plot(iterations, chr22a_error, color='blue', linestyle='solid')
plt.plot(iterations, chr22a_avg_error, color='blue', linestyle='dashed')

plt.plot(iterations, wil50_error, color="orange", linestyle='solid')
plt.plot(iterations, wil50_avg_error, color="orange", linestyle='dashed')

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

ax.legend(["esc32a",
           "esc32a",
           "ste36a",
           "ste36a",
           "chr22a",
           "chr22a",
           "wil50",
           "wil50"], loc='center left', bbox_to_anchor=(1, 0.5))

plt.xlabel("Permutacje/blok")
plt.ylabel("Błąd względny [%]")
plt.axis((100, 1024, 0, 70))
plt.grid()
plt.show()