import time

import numpy as np
import progressbar

for i in progressbar.progressbar(np.arange(100)):
    time.sleep(0.02)


with progressbar.ProgressBar(max_value=10) as bar:
    for i in range(10):
        time.sleep(0.1)
        bar.update(i)
