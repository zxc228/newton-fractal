import numpy as np
import cmath
import matplotlib.pyplot as plt
from math import sqrt
from numba import njit
from random import randint

W = 1000
H = 500
X_0 = W // 2
Y_0 = H // 2


@njit
def raschet_1():
    img1 = np.zeros((H, W))
    img2 = np.zeros((800, 800))
    for i in range(H):
        for j in range(W):
            x = (i - Y_0) * 0.1
            y = (j - X_0) * 0.1
            z = complex(x, y)
            if x or y:
                t = complex(0, 0)
                while sqrt((z.real - t.real) ** 2 + (z.imag - t.imag) ** 2) >= 0.01:
                    t = z
                    z = 0.8 * z + 0.2 * pow(z, -4)
                res = int(cmath.phase(z) / 0.628318530717959)
                color = 0

                if res == 0:
                    color = randint(10, 15)
                elif res == 2 or res == 1:
                    color = randint(6, 8)
                elif res == 4 or res == 3:
                    color = randint(4, 5)
                elif res == -4 or res == -3:
                    color = randint(20, 25)
                elif res == -2 or res == -1:
                    color = randint(1, 2)
                img1[i][j] = color

    ma = 1000000
    mi = 1 / ma
    for y in range(-400, 400):
        for x in range(-400, 400):
            n = 0
            z = complex(x * 0.005, y * 0.005)
            d = z
            while (z.real) ** 2 + (z.imag) ** 2 < ma and (d.real) ** 2 + (d.imag) ** 2 > mi and n < 50:
                t = z
                p = (((t.real) ** 2 + (t.imag) ** 2)) ** 2
                x_z = 2 / 3 * t.real + ((t.real) ** 2 - (t.imag) ** 2) / (3 * p)
                z = complex(x_z, z.imag)
                y_z = 2 / 3 * t.imag * (1 - t.real / p)
                z = complex(z.real, y_z)
                x_d = abs(t.real - z.real)
                d = complex(x_d, d.imag)
                y_d = abs(t.imag - z.imag)
                d = complex(d.real, y_d)
                n += 1
            color = (n * 9) % 255
            img2[400 + x][400 + y] = color

    return img1, img2


img = raschet_1()

ax = plt.subplot(1, 2, 1)
ax.imshow(img[0])

ax = plt.subplot(1, 2, 2)
ax.imshow(img[1])

plt.show()
