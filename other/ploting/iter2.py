import numpy as np
import matplotlib.pyplot as plt

# plt.style.use('seaborn-whitegrid')

iterations = [10, 25, 50, 75, 100, 200, 300, 400, 500, 1000, 2000]

chr22a_error = [82, 181, 311, 458, 611, 1204, 1880, 1624, 2575, 4920, 9354]

esc32a_error = [137, 313, 575, 827, 1036, 1951, 2894, 3851, 4765, 9209, 18104]

ste36a_error = [161, 416, 696, 981, 1267, 2416, 3530, 4683, 5821, 5821, 22653]

wil50_error = [262, 616, 1093, 1594, 2067, 4014, 5965, 7930, 10082, 20092, 39258]

fig = plt.figure(figsize=(10, 6))
ax = plt.axes()


plt.plot(iterations, esc32a_error, color="green", linestyle='solid')
plt.plot(iterations, ste36a_error, color="red", linestyle='solid')
plt.plot(iterations, chr22a_error, color='blue', linestyle='solid')
plt.plot(iterations, wil50_error, color="orange", linestyle='solid')


ax.legend(["esc32a", "ste36a", "chr22a", "wil50"])

plt.xlabel("Liczba iteracji")
plt.ylabel("Czas obliczeń [ms]")
plt.axis((10, 2000, 0, 40000))
plt.grid()
plt.show()