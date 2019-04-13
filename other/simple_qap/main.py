import itertools

distances = [
    [0, 50, 50, 94, 50],
    [50, 0, 22, 50, 36],
    [50, 22, 0, 44, 14],
    [94, 50, 44, 0, 50],
    [50, 36, 14, 50, 0]
]

flow = [
    [0, 0, 2, 0, 3],
    [0, 0, 0, 3, 0],
    [2, 0, 0, 0, 0],
    [0, 3, 0, 0, 1],
    [3, 0, 0, 1, 0]
]


def qap(permutation):
    result = 0

    for a in range(len(permutation)):
        for b in range(len(permutation)):
            weight = flow[a][b]
            first_point = permutation[a] - 1
            second_point = permutation[b] - 1
            distance = distances[first_point][second_point]

            result += weight * distance

    return result

permutations = itertools.permutations([1, 2, 3, 4, 5])

i = 0

for p in permutations:
    print(p, " & ", qap(p), end=" ")
    i += 1

    if i != 4:
        print(" & ", end="")
    else:
        print("\\\\")
        i = 0
