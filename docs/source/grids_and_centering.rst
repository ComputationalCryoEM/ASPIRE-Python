Grids and Centering
===================

Center of a discrete image
#########

The center pixel of a zero-indexed sequence of length :math:`n` is defined to be :math:`\frac{n}{2}` if :math:`n` is even and :math:`\frac{n-1}{2}` if :math:`n` is odd.
This convention applies when defining the index of the center pixel for a discrete image or volume. So for example, the center coordinates of an 8x8 image would be :math:`(4,4)`, while for a 7x7 image the would be :math:`(3,3)`. Note that in the case of an even resolution, the convention is to bias "to the right", i.e. towards higher index values.

Grids
#########
