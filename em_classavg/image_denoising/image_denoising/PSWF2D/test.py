from PSWF2D.PSWF2DModel import PSWF2D
from PSWF2D.GeneralFunctions import leggauss_0_1
import scipy.special as special
import numpy as np
import time


def test():
    truncation = 1e-6
    resolutions = 2 ** np.arange(7)
    beta = 1.0

    ee_o = []
    ee_i = []

    for resolution in resolutions:
        bandlimit = beta * np.pi * resolution
        big_n = max(4 * resolution, 2 ** 7)
        x, w = leggauss_0_1(big_n)
        x_eval = x * resolution
        y_eval = np.zeros(big_n)
        r_2d_grid_on_the_circle = np.sqrt(np.square(x_eval) + np.square(y_eval)) / resolution
        theta_2d_grid_on_the_circle = np.angle(x_eval + 1j * y_eval)

        print("testing for resolution = {}".format(resolution))

        tic = time.clock()
        pswf2d = PSWF2D(bandlimit)
        toc = time.clock()
        print("finished initializing the model in {} seconds".format(toc - tic))

        # find max alpha for each N
        max_ns = []
        a = np.square(float(beta * resolution) / 2)
        m = 0
        alpha_all = []
        while True:
            alpha = pswf2d.alpha_all[m]

            lambda_var = a * np.square(np.absolute(alpha))
            gamma = np.sqrt(np.absolute(lambda_var / (1 - lambda_var)))

            n_end = np.where(gamma <= truncation)[0]

            if len(n_end) != 0:
                n_end = n_end[0]
                if n_end == 0:
                    break
                max_ns.extend([n_end])
                alpha_all.extend(alpha[:n_end])
                m += 1

        angular_frequency = np.repeat(np.arange(len(max_ns)), max_ns).astype('float')
        alpha_nn = np.array(alpha_all)

        tic = time.clock()
        samples = pswf2d.evaluate_all(r_2d_grid_on_the_circle, theta_2d_grid_on_the_circle, max_ns)
        toc = time.clock()
        print("finished evaluating points in {} seconds\n".format(toc - tic))

        t = x * w
        e_o = 0
        a = bandlimit * np.outer(x, x)
        full_r = samples * alpha_nn / (2 * np.pi)
        e_i = 0
        for i in range(int(np.max(angular_frequency)) + 1):

            # testing orthonormality of radial part
            temp_mat = samples[:, angular_frequency == i]
            temp = 2 * np.pi * np.dot(temp_mat.T * t, temp_mat)
            e_o = max(e_o, np.max(np.absolute(temp - np.identity(len(temp)))))

            # test radial integration equation
            temp = special.jv(i, a) * t
            temp = np.dot(temp, temp_mat)
            r = full_r[:, angular_frequency == i] / (1j ** i)
            e_i = max(e_i, np.max(np.absolute(temp - r)))

        print("orthogonal test error: {}".format(e_o))
        print("radial integration equation test error: {}\n".format(e_i))

        ee_o.append(np.max(e_o))
        ee_i.append(np.max(e_i))

    print('errors in orthonormality test:')
    print(ee_o)
    print('errors in integral equation test:')
    print(ee_i)


test()
