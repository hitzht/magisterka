from typing import Any



def himmelblau(*point) -> float:
    if len(point) == 2:
        x = point[0]
        y = point[1]
    elif len(point) == 1:
        x = point[0][0]
        y = point[0][1]
    else:
        raise ValueError("Invalid himmelblau function argument")

    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


print(himmelblau([0,0]))
print(himmelblau(0, 0))
