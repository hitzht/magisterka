import numpy as np
import matplotlib.pyplot as plt

# plt.style.use('seaborn-whitegrid')

iterations = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1024]

chr22a_error = [146, 330, 753, 1063, 1462, 2009, 2664, 3241, 3769, 4825]

esc32a_error = [249, 590, 1178, 1664, 2370, 3363, 4388, 5172, 6131, 7641]

ste36a_error = [323, 750, 1359, 2106, 2978, 4106, 5401, 5401, 7622, 9246]

wil50_error = [525, 1225, 2042, 3482, 4676, 6315, 8335, 9906, 11323, 13494]

fig = plt.figure(figsize=(10, 6))
ax = plt.axes()


plt.plot(iterations, esc32a_error, color="green", linestyle='solid')
plt.plot(iterations, ste36a_error, color="red", linestyle='solid')
plt.plot(iterations, chr22a_error, color='blue', linestyle='solid')
plt.plot(iterations, wil50_error, color="orange", linestyle='solid')


ax.legend(["esc32a", "ste36a", "chr22a", "wil50"])

plt.xlabel("Permutacje/blok")
plt.ylabel("Czas obliczeń [ms]")
plt.axis((100, 1024, 0, 15000))
plt.grid()
plt.show()