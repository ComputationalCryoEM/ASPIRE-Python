"""
Test Gallery
============

This is a test gallery. Hello sphinx.
"""

# %%
# Placeholder Plot
# ----------------
#
# Here is a pretty plot.

# libraries
import matplotlib.pyplot as plt
import numpy as np

# create data
x = np.random.normal(size=50000)
y = x * 3 + np.random.normal(size=50000)

# Big bins
plt.hist2d(x, y, bins=(50, 50), cmap=plt.cm.jet)
plt.show()

# Small bins
plt.hist2d(x, y, bins=(300, 300), cmap=plt.cm.jet)
plt.show()

# If you do not set the same values for X and Y, the bins won't be a square!
plt.hist2d(x, y, bins=(300, 30), cmap=plt.cm.jet)
plt.show()
