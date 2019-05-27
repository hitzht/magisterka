import numpy as np
import matplotlib.pyplot as plt

# plt.style.use('seaborn-whitegrid')

iterations = [10, 25, 50, 75, 100, 150, 200, 250, 300]

chr22a_error = [256, 580, 1104, 1514, 2338, 3161, 3976, 4866, 5974]

esc32a_error = [498, 1072, 2028, 2958, 4113, 6099, 7872, 9746, 11709]

ste36a_error = [539, 1319, 2568, 3775, 5058, 7465, 9955, 12614, 15458]

wil50_error = [938, 2380, 4692, 6955, 9051, 13711, 18202, 22329, 26766]

fig = plt.figure(figsize=(10, 6))
ax = plt.axes()


plt.plot(iterations, esc32a_error, color="green", linestyle='solid')
plt.plot(iterations, ste36a_error, color="red", linestyle='solid')
plt.plot(iterations, chr22a_error, color='blue', linestyle='solid')
plt.plot(iterations, wil50_error, color="orange", linestyle='solid')


ax.legend(["esc32a", "ste36a", "chr22a", "wil50"])

plt.xlabel("Liczba bloków")
plt.ylabel("Czas obliczeń [ms]")
plt.axis((10, 300, 0, 27000))
plt.grid()
plt.show()