from __future__ import absolute_import
from __future__ import print_function

def gen_coordinates(lattice_vec=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    bas=[0, 0, 0], ncell=[1, 1, 1], block_size=0):
    import numpy as np
    import itertools as itr

    tol = 1e-6

    invmatrix = np.zeros([3, 3], dtype=np.float64)
    detmatrix = np.linalg.det(lattice_vec)
    if (abs(detmatrix) > tol):
        invmatrix = np.linalg.inv(lattice_vec)

    icvec = np.zeros([3], dtype=np.float64)
    bsf = np.zeros([3], dtype=np.float64)
    for I0 in range(0, len(bas)):
        icvec[0] = np.sum(bas[I0, :]*invmatrix[:, 0])
        icvec[1] = np.sum(bas[I0, :]*invmatrix[:, 1])
        icvec[2] = np.sum(bas[I0, :]*invmatrix[:, 2])
        bsf[:] = np.floor(icvec[:] + 1e-7)
        for mu in range(0, 3):
            bas[I0, mu] = bas[I0, mu] - np.sum(bsf[:]*lattice_vec[:, mu])

    ii = 0
    coord = np.zeros([len(bas)*np.prod(ncell), 3], dtype=np.float64)
    for II3, II2, II1 in itr.product(range(0, ncell[2], block_size),
                                     range(0, ncell[1], block_size),
                                     range(0, ncell[0], block_size)):
        for I3, I2, I1, I0 in\
            itr.product(range(II3, min(II3 + block_size, ncell[2])),
                        range(II2, min(II2 + block_size, ncell[1])),
                        range(II1, min(II1 + block_size, ncell[0])),
                        range(0, len(bas))):
            coord[ii, :] = I1*lattice_vec[0, :] + I2*lattice_vec[1, :]\
                + I3*lattice_vec[2, :] + bas[I0, :]
            ii = ii + 1

    return coord

def get_num_atoms(num_unit_cell=1, ncell=[1, 1, 1]):
    import numpy as np
    return num_unit_cell*np.prod(ncell)