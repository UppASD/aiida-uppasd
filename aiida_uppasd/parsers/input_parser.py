from __future__ import absolute_import

from aiida.engine import ExitCode
from aiida.orm import Dict, XyData
from aiida.parsers.parser import Parser

class ASDInputParser(object):

    def __init__(self, node):
        """[summary]
        
        Arguments:
            node {[type]} -- [description]
        """
        return  

    def print_inpsd(self, ASDInputs=None):
        """[summary]
        
        Keyword Arguments:
            ASDInputs {[type]} -- [description] (default: {None})
        """

        if ASDInputs is not None:
            # Write the inpsd.dat to file
            inpsd_file = open('inpsd.dat', 'w')
            for name in ASDInputs:
                for descriptor in ASDInputs[name]:
                    current = ASDInputs[name][descriptor]
                    if isinstance(current, list):
                        if len(descriptor) > 0:
                            inpsd_file.write('{descriptor}  '
                                             .format(**locals()))
                        for ii in range(len(current)):
                            line = current[ii]
                            if isinstance(line, list):
                                for jj in range(len(line)):
                                    entry = line[jj]
                                    inpsd_file.write('{entry}  '
                                                     .format(**locals()))
                                inpsd_file.write('\n')
                            else:
                                inpsd_file.write('{line}  '.format(**locals()))
                        inpsd_file.write('\n')
                    elif isinstance(current, tuple):
                        inpsd_file.write('{descriptor} '.format(**locals()))
                        for ii in range(len(current)):
                            entry = current[ii]
                            inpsd_file.write('{entry}  '.format(**locals()))
                        inpsd_file.write('\n')
                    else:
                        inpsd_file.write('{descriptor}  {current}\n'
                                         .format(**locals()))
                inpsd_file.write('\n')
            inpsd_file.close()
        else:
            print("Error")
        return

    def print_momfile(self, chem_type=[None], mag_mom=[None], mom=[None]):
        """[summary]
        
        Keyword Arguments:
            chem_type {list} -- [description] (default: {[None]})
            mag_mom {list} -- [description] (default: {[None]})
            mom {list} -- [description] (default: {[None]})
        """

        if mag_mom is not None:
            mom_file = open('momfile.dat', 'w')
            _mom_fmt = "{: 4d} {: 4d} {: 4.6f} {: 4.6f} {: 4.6f} {: 4.6f}\n"
            for iat in range(len(chem_type)):
                mom_file.write(_mom_fmt.format(iat + 1, chem_type[iat],
                                               mag_mom[iat], mom[iat, 0],
                                               mom[iat, 1], mom[iat, 2]))
            mom_file.close()
        else:
            print("Error")
        return

    def print_posfile(self, bas=[None], chem_type=[None], atom_type=[None],
                      chem_conc=[None], alloy=False):
        """[summary]
        
        Keyword Arguments:
            bas {list} -- [description] (default: {[None]})
            chem_type {list} -- [description] (default: {[None]})
            atom_type {list} -- [description] (default: {[None]})
            chem_conc {list} -- [description] (default: {[None]})
            alloy {bool} -- [description] (default: {False})
        """

        if bas is not None:

            pos_file = open('posfile.dat', 'w')
            if alloy:
                _pos_fmt = "{: 4d} {: 4d} {: 4d} {: 4.6f}"\
                    + "{: 4.6f} {: 4.6f} {: 4.6f}\n"
                for iat in range(len(atom_type)):
                    pos_file.write(_pos_fmt.format(iat + 1, atom_type[iat],
                                                   chem_type[iat],
                                                   chem_conc[iat],
                                                   bas[iat, 0],
                                                   bas[iat, 1],
                                                   bas[iat, 2]))
            else:
                _pos_fmt = "{: 4d} {: 4d} {: 4.6f} {: 4.6f} {: 4.6f}\n"
                for iat in range(len(atom_type)):
                    pos_file.write(_pos_fmt.format(iat + 1, atom_type[iat],
                                                   bas[iat, 0],
                                                   bas[iat, 1],
                                                   bas[iat, 2]))

            pos_file.close()
        else:
            print("Error")
        return

    def print_kfile(self, ani_type=None, ani_val_k1=[None], ani_val_k2=[None],
                    ani_dir=[None], ani_rat=[None]):
        """[summary]
        
        Keyword Arguments:
            ani_type {[type]} -- [description] (default: {None})
            ani_val_k1 {list} -- [description] (default: {[None]})
            ani_val_k2 {list} -- [description] (default: {[None]})
            ani_dir {list} -- [description] (default: {[None]})
            ani_rat {list} -- [description] (default: {[None]})
        """

        if ani_type.lower() == 'uniaxial':
            ani_type = [1]*len(ani_val_k1)
        elif ani_type.lower() == 'cubic':
            ani_type = [2]*len(ani_val_k1)
        elif ani_type.lower() == 'combined':
            ani_type = [7]*len(ani_val_k1)
        else:
            pass

        if ani_type is not None:
            ani_file = open('kfile.dat' , 'w')
            _ani_fmt = ("{: 4d} {: 4d} {: 4.6f} {: 4.6f}" 
                        "{: 4.6f} {: 4.6f} {: 4.6f} {: 4.6f}\n")
            for iat in range(len(ani_val_k1)):
                ani_file.write(_ani_fmt.format(iat + 1, ani_type[iat],
                                               ani_val_k1[iat],
                                               ani_val_k2[iat],
                                               ani_dir[iat, 0],
                                               ani_dir[iat, 1],
                                               ani_dir[iat, 2],
                                               ani_rat[iat]))
            ani_file.close()

        return

    def print_restart(self, num_atoms=1, num_ens=1, mom_mag=[None], mag=[None],
                      restart_name="restart._UppASD_.out"):
        """[summary]
        
        Keyword Arguments:
            num_atoms {int} -- [description] (default: {1})
            num_ens {int} -- [description] (default: {1})
            mom_mag {list} -- [description] (default: {[None]})
            mag {list} -- [description] (default: {[None]})
            restart_name {str} -- [description] (default: {"restart._UppASD_.out"})
        """
        import itertools as itr

        if mag is not None:
            restart_fmt =\
                "{:8d}{:8d}{:8d}  {: 16.8E}{: 16.8E}{: 16.8E}{: 16.8E}\n"

            restart_file = open(restart_name, 'w')

            def restart_header(restart_file, num_atoms=1, num_ens=1):

                restart_file.write("{:s}\n".format('#'*80))
                restart_file.write("{:s}\n".format('# File type: R'))
                restart_file.write("{:s}\n".format('# Simulation type: Init'))
                restart_file.write("{:s} {:8d}\n".format('# Number of atoms: ',
                                                         num_atoms))
                restart_file.write("{:s} {:8d}\n".\
                    format('# Number of ensembles: ', num_ens))
                restart_file.write("{:s}\n".format('#'*80))
                restart_file.\
                    write("{:>8s}{:>8s}{:>8s}{:>16s}{:>16s}{:>16s}{:>16s}\n".
                          format("#iter", "ens", "iatom", "|Mom|",
                                 "M_x", "M_y", "M_z"))
                return

            for ens, iat in itr.product(range(num_ens), range(num_atoms)):
                restart_file.write(restart_fmt.format(-1, ens + 1, iat + 1,
                                   mom_mag[iat], mag[ens, iat, 0],
                                   mag[ens, iat, 1], mag[ens, iat, 2]))
            restart_file.close()

        return

    def print_exchange(self, atom_type=[None], chem_type=[None],
                       bond_vec=[None], exc_int=[None], alloy=False,
                       maptype=2):
        """[summary]
        
        Keyword Arguments:
            atom_type {list} -- [description] (default: {[None]})
            chem_type {list} -- [description] (default: {[None]})
            bond_vec {list} -- [description] (default: {[None]})
            exc_int {list} -- [description] (default: {[None]})
            alloy {bool} -- [description] (default: {False})
            maptype {int} -- [description] (default: {2})
        """
        import numpy as np

        if exc_int is not None:
            exchange_file = open("jfile.dat", "w")
            if alloy:
                if maptype == 1:
                    _jij_fmt = (" {: 4d} {: 4d} {: 4d} {: 4d}"
                                " {: 4.6f} {: 4.6f} {: 4.6f} {: 4.6f}"
                                " {: 4.6f}\n")
                elif maptype == 2:
                    _jij_fmt = (" {: 4d} {: 4d} {: 4d} {: 4d}"
                                " {: 4d} {: 4d} {: 4d} {: 4.6f} {: 4.6f}\n")
                else:
                    print("Error")

                for iat in range(len(exc_int)):
                    exchange_file.\
                        write(_jij_fmt.format(atom_type[iat, 0],
                                              atom_type[iat, 1],
                                              chem_type[iat, 0],
                                              chem_type[iat, 1],
                                              bond_vec[iat, 0],
                                              bond_vec[iat, 1],
                                              bond_vec[iat, 2],
                                              exc_int[iat],
                                              np.linalg.norm(bond_vec[iat])))
            else:
                if maptype == 1:
                    _jij_fmt = (" {: 4d} {: 4d} {: 4.6f} {: 4.6f} {: 4.6f}"
                                " {: 4.6f} {: 4.6f}\n")
                elif maptype == 2:
                    _jij_fmt = (" {: 4d} {: 4d} {: 4d} {: 4d} {: 4d}"
                                " {: 4.6f} {: 4.6f}\n")

                for iat in range(len(exc_int)):
                    exchange_file.\
                        write(_jij_fmt.format(atom_type[iat, 0],
                                              atom_type[iat, 1],
                                              bond_vec[iat, 0],
                                              bond_vec[iat, 1],
                                              bond_vec[iat, 2],
                                              exc_int[iat],
                                              np.linalg.norm(bond_vec[iat])))
            exchange_file.close()
        else:
            print("Error")
        return

    def print_dm_vector(self, atom_type=[None], chem_type=[None],
                       bond_vec=[None], dm_vec=[None], alloy=False, maptype=2):
        """[summary]
        
        Keyword Arguments:
            atom_type {list} -- [description] (default: {[None]})
            chem_type {list} -- [description] (default: {[None]})
            bond_vec {list} -- [description] (default: {[None]})
            dm_vec {list} -- [description] (default: {[None]})
            alloy {bool} -- [description] (default: {False})
            maptype {int} -- [description] (default: {2})
        """
        import numpy as np

        if dm_vec is not None:
            dm_file = open("dmfile.dat", "w")
            if alloy:
                if maptype == 1:
                    _dij_fmt = (" {: 4d} {: 4d} {: 4d} {: 4d}"
                                " {: 4.6f} {: 4.6f} {: 4.6f}"
                                " {: 4.6f} {: 4.6f} {: 4.6f} {: 4.6f}\n")
                elif maptype == 2:
                    _dij_fmt = (" {: 4d} {: 4d} {: 4d} {: 4d}"
                                " {: 4d} {: 4d} {: 4d}"
                                " {: 4.6f} {: 4.6f} {: 4.6f} {: 4.6f}\n")
                else:
                    print("Error")
                for iat in range(len(dm_vec)):
                    dm_file.\
                        write(_dij_fmt.format(atom_type[iat, 0],
                                              atom_type[iat, 1],
                                              chem_type[iat, 0],
                                              chem_type[iat, 1],
                                              bond_vec[iat, 0],
                                              bond_vec[iat, 1],
                                              bond_vec[iat, 2],
                                              dm_vec[iat, 0],
                                              dm_vec[iat, 1],
                                              dm_vec[iat, 2],
                                              np.linalg.norm(bond_vec[iat])))
            else:
                if maptype == 1:
                    _dij_fmt = (" {: 4d} {: 4d} {: 4.6f} {: 4.6f} {: 4.6f}"
                                " {: 4.6f} {: 4.6f} {: 4.6f} {: 4.6f}\n")
                elif maptype == 2:
                    _dij_fmt = (" {: 4d} {: 4d} {: 4d} {: 4d} {: 4d}"
                                " {: 4.6f} {: 4.6f} {: 4.6f} {: 4.6f}\n")
                else:
                    print("Error")
                for iat in range(len(dm_vec)):
                    dm_file.\
                        write(_dij_fmt.format(atom_type[iat, 0],
                                              atom_type[iat, 1],
                                              bond_vec[iat, 0],
                                              bond_vec[iat, 1],
                                              bond_vec[iat, 2],
                                              dm_vec[iat, 0],
                                              dm_vec[iat, 1],
                                              dm_vec[iat, 2],
                                              np.linalg.norm(bond_vec[iat])))
        return
