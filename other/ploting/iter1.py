import numpy as np
import matplotlib.pyplot as plt

# plt.style.use('seaborn-whitegrid')

iterations = [10, 25, 50, 75, 100, 200, 300, 400, 500, 1000, 2000]
chr22a_error = [23, 19, 17, 13, 11, 7.3, 6.3, 6.1, 3.6, 2.6, 1.9]
chr22a_avg_error = [33, 25, 19, 16, 14, 9.1, 7.7, 7.7, 4.9, 3.6, 3.2]

esc32a_error = [91, 72, 52, 49, 43, 23, 15, 15, 11, 6.2, 4.6]
esc32a_avg_error = [110, 85, 61, 52, 45, 28, 20, 18, 14, 9.2, 6.2]

ste36a_error = [62, 51, 39, 34, 32, 20, 14, 11, 10, 4.6, 1.4]
ste36a_avg_error = [66, 55, 45, 38, 34, 23, 17, 14, 12, 6.9, 3.9]

wil50_error = [8.2, 7.1, 6.8, 5.1, 5.2, 4.1, 3.4, 2.5, 2.5, 1.6, 1]
wil50_avg_error = [9.1, 7.9, 7.1, 6.1, 5.6, 4.5, 3.6, 3, 2.8, 1.7, 1.1]


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

ax.legend(["esc32a",
           "esc32a",
           "ste36a",
           "ste36a",
           "chr22a",
           "chr22a",
           "wil50",
           "wil50"])

plt.xlabel("Liczba iteracji")
plt.ylabel("Błąd względny [%]")
plt.axis((10, 2000, 0, 70))
plt.grid()
plt.show()