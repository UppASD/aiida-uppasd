# -*- coding: utf-8 -*-
"""Utilties to convert between inpsd text format, Python dict, and JSON output"""

# Import numpy for array handling
import numpy as np
from sys import stdout,stderr
from aiida.orm import StructureData
from aiida import load_profile
from aiida.tools.data.array import kpoints
from aiida.tools import spglib_tuple_to_structure,get_kpoints_path,get_explicit_kpoints_path

def d2e(d):
    """Convert Fortran double precision exponents to Python single

    :param d: string containing exponent
    """
    e='e-'.join(d.split('d-'))
    return float(e)

def autoconvert(s):
    """ Convert values to optimal(...) type. 

    :param s: data to be converted to proper type

    Credit: Stack Exchange
    """

    for fn in (int, float,d2e,str):
        try:
            return fn(s)
        except ValueError:
            pass
    return s

def dq(s):
    """ Add double quotes around strings for JSON

    :param s: string to be decorated with double quotes
    """
    return '"%s"' % s

def json_conv(s):
    """ Convert data to string according to JSON format

    :param s: input value/list/array 
    """
    ty=type(s)
    if(ty==float or ty==int):
        data=s
    elif(ty==str):
        data=dq(s)
    elif(ty==np.ndarray):
        data=s.tolist()
    elif(ty==list):
        # Special check for lists of characters
        if(type(s[0])==str and len(s)==3):
            data='[ "'+s[0]+'" , "'+s[1]+'" , "'+s[2]+'" ]'
            #data=['"'+ss+'"' for ss in s]
        else:
            data=s
    return str(' : '+str(data)+',')


# Define keys that need more than one value
vecarg_list=['hfield',
           'ip_hfield',
           'qm_svec',
           'qm_nvec',
           'jvec',
           'bc',
           'ncell']

# Define keys that have a special list for input
special_list=['ip_nphase',
              'ip_mcanneal',
             'ntraj',
             'positions',
             'moments']

def inpsd_to_dict(f):
    """Read UppASD input to Python dict

    The function defaults to single value for every key but exceptions are
    handled by the pre-defined lists "vecarg_list" and "special_list".
    Currently all keywords except spatial temperature gradients are supported.

    :param f: file handle to read from
    """ 
    inpdict = {}
    for line in f:
        if(len(line)>1):
            key = line.split()[0].lower()
            if(key=='cell'):
                a1=line.split()[1:4];a2=f.readline().split()[0:3];a3=f.readline().split()[0:3]
                data=np.asarray([np.asarray(a1,dtype=float),
                                np.asarray(a2,dtype=float),
                                np.asarray(a3,dtype=float)])
            elif(key in special_list):
                nlines=autoconvert(line.split()[1])
                data=[]
                data.append(nlines)
                for iline in range(nlines):
                    dline=f.readline().split()
                    data.append([autoconvert(i) for i in dline])
            elif(key in vecarg_list):
                data=[ autoconvert(s) for s in line.split()[1:4]]
            else:
                data=autoconvert(line.split()[1])
            inpdict[key]=data
    return inpdict

def dict_to_inpsd(inpdict,f=stdout):
    """ Print dict as UppASD input format

    The function prints either to the screen or to a file if defined

    :param inpdict: The dict that will be printed
    :param f: file handle for writing to file (optional parameter)
    """ 

    for key in inpdict.keys():
        if(key=='cell'):
            print(key,   np.array2string(inpdict[key][0])[1:-1],file=f)
            print('    ',np.array2string(inpdict[key][1])[1:-1],file=f)
            print('    ',np.array2string(inpdict[key][2])[1:-1],file=f)
        elif(key in special_list):
            nlines=inpdict[key][0]
            print(key,nlines,file=f)
            for iline in range(nlines):
                print(" ".join([str(w) for w in inpdict[key][iline+1]]),file=f)
        elif(key in vecarg_list):
            print(key," ".join([str(w) for w in inpdict[key]]),file=f)
        else:
            print(key,inpdict[key],file=f)

def dict_to_JSON(inpdict,f=stdout):
    """ Print dict on JSON format

    Hand-written routine is preferred since the standard implementation has issues
    with numpy arrays and insists on line-breaking lists ad nauseum

    The function prints either to the screen or to a file if defined

    :param inpdict: The dict that will be printed
    :param f: file handle for writing to file (optional parameter)
    """
    print('{',file=f)
    for key in inpdict.keys():
        print('  ',dq(key),json_conv(inpdict[key]),file=f)
    print('  "comment" : "Autoconverted UppASD dict"',file=f)
    print('}',file=f)


def read_posfile(posfile):
    """ Read the position file

    :param: file name for 'posfile'
    """ 
    try:
        with open(posfile,'r') as pfile:
          lines=pfile.readlines()
          positions=np.empty([0,3])
          numbers=[]
          for idx,line in enumerate(lines):
             line_data=line.rstrip('\n').split()
             if len(line_data)>0:
                positions=np.vstack((positions,np.asarray(line_data[2:5],dtype=float)))
                numbers=np.append(numbers,np.asarray(line_data[1],dtype=float))
        return positions,numbers
    except IOError:
        print("Position file ", posfile, "not accessible.")


if __name__ == '__main__':
    
    # Read inpsd file to dict
    with open('inpsd.dat','r') as f:
        inpdict=inpsd_to_dict(f)
    
    
    # Write dict to JSON format
    with open('inpsd.json','w') as f:
        dict_to_JSON(inpdict,f)
    
    # Write dict to inpsd format to screen
    ### dict_to_inpsd(inpdict)
    
    # Setup structure for later use (spglib or aiida DataStructure)
    pos_exist=False
    if 'posfile' in inpdict.keys():
        positions,numbers = read_posfile(inpdict['posfile'])
        pos_exist=True
    elif 'positions' in inpdict.keys():
        pos_arr=np.array(inpdict['positions'][1:])
        positions = pos_arr[:,2:5]
        numbers = pos_arr[:,1]
        pos_exist=True


    if pos_exist:
        cell=inpdict['cell']
        pbc=[]
        for entry in inpdict['bc']:
            pbc.append(entry=='P')
        print('pbc:',pbc)
        spgcell=(cell,positions,numbers)
        #spacegroup = spg.get_spacegroup(spgcell, symprec=1e-5)
        load_profile()
        ### structure = StructureData(cell=cell)
        ### for pos in positions:
        ###     structure.append_atom(position=pos,symbols='Fe')
        ### print('----------')
        ### print(structure.cell)
        ### print(structure.sites)
        ### print(structure.sites[0])
        ### print('----------')

        structure=spglib_tuple_to_structure(spgcell)
        structure.set_pbc(pbc)
        print('----------')
        print(structure.pbc)
        print(structure.cell)
        print(structure.sites)
        print('----------')
        #kp=get_explicit_kpoints_path(structure,method='legacy')
        kp=get_kpoints_path(structure,method='seekpath')
        print(kp.keys())
        print(kp['parameters']['point_coords'])
        print(kp['parameters']['path'])
        ### print(kp['primitive_structure'].cell)
        ### print(kp['primitive_structure'].sites)
        ### print(kp['explicit_kpoints'].labels)



