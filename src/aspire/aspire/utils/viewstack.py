import matplotlib.pyplot as plt

from aspire.aspire.common.logger import logger


def view_stack(stack, numslices=16, startslice=1, nrows=4, ncols=4):

    if numslices > nrows * ncols:
        raise ValueError("nrows*ncols should be at least numslices"
                         " to accomodate all images.")

    logger.debug("Shape of loaded map: " + str(stack.shape))

    plt.figure(figsize=(4, 4))

    j = 0
    dx = 0.95 / ncols
    dy = 0.95 / nrows
    for col in range(ncols):
        for row in range(nrows):
            if j < numslices:
                x0 = col * 1.0 / ncols
                y0 = 1 - row * 1.0 / nrows - dy
                # Flip y direction to fill image grid from top
                ax = plt.axes([x0, y0, dx, dy])
                plt.imshow(stack[j, ...])
                plt.gray()
                plt.axis("off")
                ax.text(0.1, 0.9, str(j + startslice), fontsize=10,
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax.transAxes, color='red',
                        bbox=dict(facecolor='yellow', alpha=0.15))

                j += 1
    plt.show()
