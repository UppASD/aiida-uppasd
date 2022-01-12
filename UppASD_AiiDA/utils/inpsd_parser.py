# -*- coding: utf-8 -*-
"""Utilties to convert between inpsd text format, Python dict, and JSON output"""

# Import numpy for array handling
import numpy as np
from sys import stdout,stderr

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
        data=s
    return str(': '+str(data)+',')


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
             'ntraj']

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
        print(dq(key),json_conv(inpdict[key]),file=f)
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
    dict_to_inpsd(inpdict,f)
    
    # Setup structure for later use (spglib or aiida DataStructure)
    if 'posfile' in inpdict.keys():
        positions,numbers = read_posfile(inpdict['posfile'])
        lattice=inpdict['cell']
        spgcell=(lattice,positions,numbers)
        #spacegroup = spg.get_spacegroup(spgcell, symprec=1e-5)


