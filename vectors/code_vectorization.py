import numpy as np
import pandas as pd


v1 = np.array(list(range(5)))
v2 = pd.Series(reversed(range(5)))

print(v1)
#v1 = [0, 1, 2, 3, 4]
#v2 = [4, 3, 2, 1, 0]

# Slowest version (loop-based)
slow_answer = sum([4.2 * (x1 * x2) for x1, x2 in zip(v1, v2)])

# Slightly faster
faster_answer = sum(4.2 * v1 * v2)

# Fastest (fully vectorized with dot product)
fastest_answer = 4.2 * v1.dot(v2)
print(fastest_answer)

total = 0
for x1, x2 in zip(v1, v2):
    total += 4.2 * (x1 * x2)

print(total)
#zip(v1, v2) → [(0,4), (1,3), (2,2), (3,1), (4,0)]
#x1 * x2 This multiply each pair.
#4.2 * (v1[0]*v2[0] + v1[1]*v2[1] + ... + v1[n]*v2[n])