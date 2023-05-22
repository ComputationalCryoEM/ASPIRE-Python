import time

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
from numpy.linalg import norm
from numpy.random import normal

from aspire.utils.bot_align import align_BO, get_angle
from aspire.utils.rotation import Rotation
from aspire.volume import Volume


def generate_data(data_name, inv_SNR):
    with mrcfile.open(data_name) as mrc:
        template = Volume(mrc.data)
    L = template.shape[1]
    shape = (L, L, L)
    ns_std = np.sqrt(inv_SNR * norm(template) ** 2 / L**3)
    vol0 = template + np.float32(normal(0, ns_std, shape))
    r = Rotation.generate_random_rotations(1)
    R_true = r._matrices[0]
    vol_given = template.rotate(r) + np.float32(normal(0, ns_std, shape))

    return vol0, vol_given, L, R_true


"""
specify the test volume and inverse SNR
"""
data_name = "emd-3683.mrc"
inv_SNR = 0


"""
The BOTalign algorithm takes in four parameters:
(1) loss type ('wemd' or 'eu')
(2) downsampling level (32 or 64 recommended)
(3) total number of iterations (150 or 200 recommended)
(4) whether refinement is performed

Below we compare the performance for four combinations of the parameters.
The experiments are repeated ntrial times and the results will be shown as boxplots.
"""
npara = 4
para = [None] * npara
ntrial = 1
para[0] = ["wemd", 32, 200, True]
para[1] = ["wemd", 64, 150, True]
para[2] = ["eu", 32, 200, True]
para[3] = ["eu", 64, 150, True]


angle_init = np.zeros((npara, ntrial))
angle_rec = np.zeros((npara, ntrial))
comp_time = np.zeros((npara, ntrial))

for trial in range(ntrial):
    print(trial)
    [vol0, vol_given, L, R_true] = generate_data(data_name, inv_SNR)

    for n in range(npara):
        tic = time.perf_counter()
        [R_init, R_rec] = align_BO(vol0, vol_given, para[n])
        toc = time.perf_counter()
        comp_time[n, trial] = toc - tic
        # Recovery without refinement (degrees)
        angle_init[n, trial] = get_angle(R_init, R_true.T)
        # Recovery with refinement (degrees)
        angle_rec[n, trial] = get_angle(R_rec, R_true.T)


"""
plot the results
"""
fig = plt.figure()
ax = fig.add_subplot(111, frameon=False)
ax.spines["top"].set_color("none")
ax.spines["bottom"].set_color("none")
ax.spines["left"].set_color("none")
ax.spines["right"].set_color("none")
ax.tick_params(labelcolor="w", top=False, bottom=False, left=False, right=False)
fig.supylabel("Rotation recovery error (degrees)")
fig.supxlabel("Average run time (seconds)")


ax1 = fig.add_subplot(121)
ax1.set_title("WEMD loss")
ax1.boxplot(
    angle_rec[0],
    positions=[0],
    notch=True,
    widths=0.5,
    patch_artist=True,
    boxprops=dict(facecolor="C0"),
)
ax1.boxplot(
    angle_rec[1],
    positions=[1],
    notch=True,
    widths=0.5,
    patch_artist=True,
    boxprops=dict(facecolor="C1"),
)
ax1.set_xticklabels(["%1.1f" % np.mean(comp_time[0]), "%1.1f" % np.mean(comp_time[1])])
ax1.xaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
ax1.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
plt.ylim([-0.3, 2])

ax2 = fig.add_subplot(122)
ax2.set_title("$L^2$ loss")
ax2.boxplot(
    angle_rec[0],
    positions=[0],
    notch=True,
    widths=0.5,
    patch_artist=True,
    boxprops=dict(facecolor="C0"),
)
ax2.boxplot(
    angle_rec[1],
    positions=[1],
    notch=True,
    widths=0.5,
    patch_artist=True,
    boxprops=dict(facecolor="C1"),
)
ax2.set_xticklabels(["%1.1f" % np.mean(comp_time[2]), "%1.1f" % np.mean(comp_time[3])])
ax2.xaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
ax2.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
plt.ylim([-0.3, 2])
plt.tick_params(
    axis="y",
    labelcolor="none",
    which="both",
    top=False,
    bottom=False,
    left=False,
    right=False,
)

plt.show()
