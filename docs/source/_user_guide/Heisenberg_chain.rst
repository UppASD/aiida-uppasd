Heisenberg chain
============================

First you need to import some package that need for calcuation:


.. code-block::

   from aiida.plugins import DataFactory, CalculationFactory
    from aiida.engine import run, workfunction
    from aiida.orm import Code, SinglefileData, Int, Float, Str, Bool, List, Dict, ArrayData, XyData, SinglefileData, FolderData, RemoteData
    import numpy as np
    import aiida
    import os
    from aiida.engine import submit
    from itertools import permutations
    import random
    aiida.load_profile()

Then you need to write your functions for single calculation:

.. code-block::
    

    def submit_single_calc(Jij1,Jij2,atom_n,ipm):
        code = Code.get_from_string('your code name')
        aiida_uppasd = CalculationFactory('UppASD_core_calculations')
        builder = aiida_uppasd.get_builder()



        #pre-prepared files folder:
        prepared_file_folder = Str(os.path.join(os.getcwd(),'HeisC_input'))
        except_filenames = List(list = [])




        # inpsd.dat file selection
        inpsd_dict = {
            'simid': Str('HeisWire'),
            'ncell': Str(f'1         1         {atom_n}'),
            'BC': Str('0         0         P '),
            'cell': Str('''1.00000 0.00000 0.00000
                    0.00000 1.00000 0.00000
                    0.00000 0.00000 1.00000'''),
            'do_prnstruct':      Int(1),
            'Mensemble': Int(1),
            'Initmag': Int(1),
            'ip_mode': Str(f'{ipm}'),
            'ip_nphase':Str('''2
                    50 1.0e-3 1e-15 0.5
                    50 0.0e-3 1e-15 0.5'''),

            'mode': Str('S'),
            'temp': Float(1.0e-3),
            'damping': Float(0.0001),
            'Nstep': Int(15000),
            'timestep': Str('1.000e-15'),
            'qpoints': Str('F'),
            'plotenergy': Int(1),
            'do_avrg': Str('Y'),
            #new added flags
            'do_tottraj':Str('Y'),
            'tottraj_step': Int(20),
        }

        exchange = {
            '1':Str(f'1 1    0.0    0.0    1.0     {Jij1}'),
            '2':Str(f'1 1    0.0    0.0    2.0     {Jij2}'),
        }





        r_l = List(list=[('*.out','.', 0)])  
        #we could use this to retrived all .out file to aiida


        # set up calculation
        inpsd_dict = Dict(dict=inpsd_dict)
        exchange_dict = Dict(dict=exchange)
        builder.code = code
        builder.prepared_file_folder = prepared_file_folder
        builder.except_filenames = except_filenames
        builder.inpsd_dict = inpsd_dict
        builder.exchange = exchange_dict
        builder.retrieve_list_name = r_l
        builder.metadata.options.resources = {'num_machines': your cluster nodes,'num_mpiprocs_per_machine': your cores}
        builder.metadata.options.max_wallclock_seconds = 60*55
        builder.metadata.options.input_filename = 'inpsd.dat'
        builder.metadata.options.parser_name = 'UppASD_core_parsers'
        builder.metadata.label = 'HeisChain'
        builder.metadata.description = 'HeisChain data'
        job_node = submit(builder)
        print('Job submitted, PK: {}'.format(job_node.pk))

then you write one simple workchain here:

.. code-block::
    @workfunction
    def HeisChain_auto(atom_number_list,ip_mode,J_ij_pair_list):
        for atom_n in atom_number_list:
            for ipm in ip_mode:
                for J_ij_pair in J_ij_pair_list:
                    Jij1 = J_ij_pair[0]/10
                    Jij2 = J_ij_pair[1]/10
                    submit_single_calc(Jij1,Jij2,atom_n,ipm)


    atom_number_list = List(list=[random.randrange(10, 50, 1) for i in range(10)])
    ip_mode = List(list=['Y','N'])
    j_seed = list(permutations([random.randrange(-10, 10, 1) for i in range(10)],2))
    if (0,0) in j_seed:
        j_seed.remove((0,0))
    J_ij_pair_list = List(list =j_seed)
        
    HeisChain_auto(atom_number_list,ip_mode,J_ij_pair_list)

Then you could visualize your result like:

..  youtube:: NeUzDa7Peu4
    :width: 640
    :height: 480

..  youtube:: ZjQBMhoTGuU
    :width: 640
    :height: 480