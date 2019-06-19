from __future__ import absolute_import
from __future__ import print_function

def gen_coordinates(lattice_vec=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    bas=[0, 0, 0], ncell=[1, 1, 1], block_size=0):
    """[summary]
    
    Keyword Arguments:
        lattice_vec {list} -- [description] (default: {[[1, 0, 0], [0, 1, 0], [0, 0, 1]]})
        bas {list} -- [description] (default: {[0, 0, 0]})
        ncell {list} -- [description] (default: {[1, 1, 1]})
        block_size {int} -- [description] (default: {0})
    
    Returns:
        [type] -- [description]
    """
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
    """[summary]
    
    Keyword Arguments:
        num_unit_cell {int} -- [description] (default: {1})
        ncell {list} -- [description] (default: {[1, 1, 1]})
    
    Returns:
        [type] -- [description]
    """
    import numpy as np
    return num_unit_cell*np.prod(ncell)

def gen_mag_configuration(num_atom_cell=1, num_ens=1, ncell=[1, 1, 1],
                          mom_cell=[None], block_size=0, conf_type='ferro',
                          coord=[None], **conf_opts):
    """[summary]
    
    Keyword Arguments:
        num_atom_cell {int} -- [description] (default: {1})
        num_ens {int} -- [description] (default: {1})
        ncell {list} -- [description] (default: {[1, 1, 1]})
        mom_cell {list} -- [description] (default: {[None]})
        block_size {int} -- [description] (default: {0})
        conf_type {str} -- [description] (default: {'ferro'})
        coord {list} -- [description] (default: {[None]})
    
    Returns:
        [type] -- [description]
    """

    if conf_type.lower() == 'ferro':
        return gen_ferro(num_atom_cell, num_ens, ncell, mom_cell, block_size,
                         **conf_opts)
    if conf_type.lower() == 'spiral':
        return gen_spiral(num_atom_cell, num_ens, ncell, mom_cell, block_size,
                          coord, **conf_opts)
    if conf_type.lower() == 'skyrmion':
        return gen_skyrmion(num_atom_cell, num_ens, ncell, mom_cell, coord,
                            block_size, **conf_opts)

def gen_ferro(num_atom_cell=1, num_ens=1, ncell=[1, 1, 1], mom_cell=[None],
              block_size=0, quantization_axis=2):
    """[summary]
    
    Keyword Arguments:
        num_atom_cell {int} -- [description] (default: {1})
        num_ens {int} -- [description] (default: {1})
        ncell {list} -- [description] (default: {[1, 1, 1]})
        mom_cell {list} -- [description] (default: {[None]})
        block_size {int} -- [description] (default: {0})
        quantization_axis {int} -- [description] (default: {2})
    
    Returns:
        [type] -- [description]
    """
    import numpy as np
    import itertools as itr

    mom = np.zeros([num_ens, num_atom_cell*np.prod(ncell), 4],
                   dtype=np.float64)

    ii = 0
    for II3, II2, II1 in itr.product(range(0, ncell[2], block_size),
                                     range(0, ncell[1], block_size),
                                     range(0, ncell[0], block_size)):
        for I3, I2, I1, I0 in\
            itr.product(range(II3, min(II3 + block_size, ncell[2])),
                        range(II2, min(II2 + block_size, ncell[1])),
                        range(II1, min(II1 + block_size, ncell[0])),
                        range(0, num_atom_cell)):
            mom[:, ii, quantization_axis] = 1.0
            mom[:, ii, 3] = mom_cell[I0]
            ii += 1
    return mom

def gen_spiral(num_atom_cell=1, num_ens=1, ncell=[1, 1, 1], mom_cell=[None],
               block_size=0, coord=[None], hl_cone_angle=0.0, hl_handness=1,
               hl_pitch_vector=[0, 0, 1], prop_vector=[0, 0, 1]):
    """[summary]
    
    Keyword Arguments:
        num_atom_cell {int} -- [description] (default: {1})
        num_ens {int} -- [description] (default: {1})
        ncell {list} -- [description] (default: {[1, 1, 1]})
        mom_cell {list} -- [description] (default: {[None]})
        block_size {int} -- [description] (default: {0})
        coord {list} -- [description] (default: {[None]})
        hl_cone_angle {float} -- [description] (default: {0.0})
        hl_handness {int} -- [description] (default: {1})
        hl_pitch_vector {list} -- [description] (default: {[0, 0, 1]})
        prop_vector {list} -- [description] (default: {[0, 0, 1]})
    
    Returns:
        [type] -- [description]
    """

    import numpy as np
    import itertools as itr

    mom = np.zeros([num_ens, num_atom_cell*np.prod(ncell), 4],
                   dtype=np.float64)
    # First do a rotation to find the rotates spin for the conical phase
    # First create a vector perpendicular to the rotation axis
    test_r = np.random.rand(3)
    # Normalize the vector
    test_r = test_r/np.sqrt(test_r.dot(test_r))
    # Axis which one will use to rotate the spins to get the conical phase
    cone_axis = np.cross(hl_pitch_vector, test_r)
    # Rotate the spin first to find the needed cone angle using Rodriges
    # rotation
    init_spin = hl_pitch_vector*np.cos(hl_cone_angle)\
        + np.cross(cone_axis, hl_pitch_vector)*np.sin(hl_cone_angle)\
        + cone_axis*(cone_axis.dot(hl_pitch_vector))*(1-np.cos(hl_cone_angle))
    # Loop over the ensembles and atoms of the system
    ii = 0
    for II3, II2, II1 in itr.product(range(0, ncell[2], block_size),
                                     range(0, ncell[1], block_size),
                                     range(0, ncell[0], block_size)):
        for I3, I2, I1, I0 in\
            itr.product(range(II3, min(II3 + block_size, ncell[2])),
                        range(II2, min(II2 + block_size, ncell[1])),
                        range(II1, min(II1 + block_size, ncell[0])),
                        range(0, num_atom_cell)):
            theta = prop_vector.dot(coord[ii, :])*2.0*np.pi*hl_handness
            # Do a Rodrigues rotation to get the helical spiral state
            mom[:, ii, :3] = init_spin*np.cos(theta)\
                + np.cross(hl_pitch_vector, init_spin)*np.sin(theta)\
                + hl_pitch_vector*(hl_pitch_vector.dot(init_spin)) *\
                    (1-np.cos(theta))
            # Normalize
            mom[:, ii, :3] = mom[:, ii, :3]/np.linalg.norm(mom[:, ii, :3]**2)
            mom[:, ii, 3] = mom_cell[I0]

    return mom

def gen_skyrmion(num_atom_cell=1, num_ens=1, ncell=[1, 1, 1], mom_cell=[None],
                 block_size=0, coord=[None], skx_orig=[0, 0, 0], skx_rad=1,
                 skx_pol=1, skx_handness=1, skx_order=1, skx_type=0):
    """[summary]
    
    Keyword Arguments:
        num_atom_cell {int} -- [description] (default: {1})
        num_ens {int} -- [description] (default: {1})
        ncell {list} -- [description] (default: {[1, 1, 1]})
        mom_cell {list} -- [description] (default: {[None]})
        block_size {int} -- [description] (default: {0})
        coord {list} -- [description] (default: {[None]})
        skx_orig {list} -- [description] (default: {[0, 0, 0]})
        skx_rad {int} -- [description] (default: {1})
        skx_pol {int} -- [description] (default: {1})
        skx_handness {int} -- [description] (default: {1})
        skx_order {int} -- [description] (default: {1})
        skx_type {int} -- [description] (default: {0})
    
    Returns:
        [type] -- [description]
    """
    import numpy as np
    import itertools as itr

    tol = 1e-6

    mom = np.zeros([num_ens, num_atom_cell*np.prod(ncell), 4],
                   dtype=np.float64)
    ii = 0
    for II3, II2, II1 in itr.product(range(0, ncell[2], block_size),
                                     range(0, ncell[1], block_size),
                                     range(0, ncell[0], block_size)):
        for I3, I2, I1, I0 in\
            itr.product(range(II3, min(II3 + block_size, ncell[2])),
                        range(II2, min(II2 + block_size, ncell[1])),
                        range(II1, min(II1 + block_size, ncell[0])),
                        range(0, num_atom_cell)):
            r_x = coord[ii, 0] - skx_orig[0]
            r_y = coord[ii, 1] - skx_orig[1]
            mod_r = np.sqrt(r_x**2 + r_y**2)
            rad = skx_rad*0.5
            r_p = (mod_r + rad)/skx_rad
            r_n = (mod_r - rad)/skx_rad
            # Out of plane angle
            theta = (np.pi + np.arcsin(np.tanh(r_p)) + np.arcsin(np.tanh(r_n))
                     + skx_pol)
            # In-plane skyrmion profile angle
            if mod_r > tol:
                phi = np.arctan2(r_y, r_x)
            else:
                phi = 0.0
            # Set the configuration
            mom[:, ii, 0] = \
                skx_handness*np.sin(theta)*np.cos(skx_order*phi + skx_type)
            mom[:, ii, 1] = \
                skx_handness*np.sin(theta)*np.sin(skx_order*phi + skx_type)
            mom[:, ii, 2] = np.cos(theta)
            # Normalize
            mom[: ,ii, :3] = mom[:, ii, :3]/np.linalg.norm(mom[:, ii, :3]**2)
            mom[:, ii, 3] = mom_cell[I0]
            ii += 1
    return mom