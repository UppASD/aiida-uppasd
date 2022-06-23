
Run with traditional UppASD input files 
========================================

A traditional UppASD input folder looks like:

::

    UppASD_folder
        |---inpsd.dat
        |---posfile
        |---momfile
        |---dmdata
        |---exchange
        |---qfile
    
Here inpsd.dat is the main input ﬁle necessary to run UppASD.
The posfile includes atomic positions,
exchange ﬁle containing the exchange interactions like Jij,
dmdata file described the DM interaction,
momfile includes the atomic moments......

The first taste of UppASD-AiiDA interface should be start with this traditional UppASD input folder directly,
with prepared folder (you could find the folder in aiida-uppasd/test_API/demo1_input) we could start writing our first caljob codes:

Firstly, we need import needed packages:

.. code-block::
    :linenos:

    from aiida.plugins import DataFactory, CalculationFactory
    from aiida.engine import run
    from aiida.orm import Code, SinglefileData, Int, Float, Str, Bool, List, Dict, ArrayData, XyData, SinglefileData, FolderData, RemoteData
    import numpy as np
    import aiida
    import os
    from aiida.engine import submit
    aiida.load_profile() #for interactive model 

then choose the code and set calculation method from UppASD_AiiDA inferface:

.. code-block::
    :linenos:

    code = Code.get_from_string('your code name')
    aiida_uppasd = CalculationFactory('uppasd.uppasd_calculation')

After that we needed to give the path to pre-prepared folder that includes all files that we need and set the except file rule. 
Note that you should named UppASD input file with "inpsd" instead of "inpsd.dat" here.

.. code-block::
    :linenos:

    prepared_file_folder = Str(os.path.join(os.getcwd(),'demo1_input'))
    except_filenames = List(list = [])

Since we want the interface collect all .out file we use:

.. code-block::
    :linenos:

    r_l = List(list=[('*.out','.', 0)])  


Finally set up calculation with your own option:

.. code-block::
    :linenos:

    builder = aiida_uppasd.get_builder()
    builder.code = code
    builder.prepared_file_folder = prepared_file_folder
    builder.except_filenames = except_filenames
    builder.retrieve_list_name = r_l
    builder.metadata.options.resources = {'num_machines': }
    builder.metadata.options.max_wallclock_seconds = 
    builder.metadata.options.parser_name = 'uppasd.uppasd_parser'
    builder.metadata.label = ''
    builder.metadata.description = ''
    job_node = submit(builder)
    print('Job submitted, PK: {}'.format(job_node.pk))

You can also find the completed code from aiida-uppasd/test_API/demo1.py 

Right now, you could submit the caljob dirctly Python or use AiiDA shell to run it interactively.
then you could check your job with：

.. code-block::
    
    verdi process show “your job PK”

if everything works greatly, you may see things like this in return:

::

    Property     Value
    -----------  ------------------------------------
    type         UppASD
    state        Finished [0]
    pk           42227
    uuid         1cda274e-1c8d-4da6-a357-e4a034d26019
    label        Demo5
    description  Test demo5 for UppASD-AiiDA
    ctime        2021-10-15 23:01:43.470045+00:00
    mtime        2021-10-15 23:02:04.939648+00:00
    computer     [20] uppasd_local

    Inputs                   PK  Type
    --------------------  -----  ------
    code                  14171  Code
    except_filenames      42225  List
    prepared_file_folder  42224  Str
    retrieve_list_name    42226  List

    Outputs           PK  Type
    -------------  -----  ----------
    averages       42236  ArrayData
    coord          42237  ArrayData
    qm_minima      42238  ArrayData
    qm_sweep       42239  ArrayData
    qpoints        42240  ArrayData
    remote_folder  42234  RemoteData
    retrieved      42235  FolderData
    totenergy      42241  ArrayData

Now you could check the result and use ASD_GUI.py to do some visualization like show the magnetic moments changes:

..  youtube:: DM5rqQ_YxyM
    :width: 640
    :height: 480







