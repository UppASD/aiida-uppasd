# -*- coding: utf-8 -*-
"""Base workchain"""
import os

from aiida import load_profile, orm
from aiida.engine import submit

from aiida_uppasd.workflows.magnon_spectra import UppASDMagnonSpectraRestartWorkflow

load_profile()
current_path = os.getcwd()

input_uppasd = {
    'inpsd_ams':
    orm.Dict(
        dict={
            'simid':
            orm.Str('bccFe100'),
            'ncell':
            orm.Str('20        20        20'),
            'BC':
            orm.Str('P         P         P '),
            'cell':
            orm.Str(
                """1.00000 0.00000 0.00000
                0.00000 1.00000 0.00000
                0.00000 0.00000 1.00000"""
            ),
            'Mensemble':
            orm.Int(1),
            'maptype':
            orm.Int(2),
            'SDEalgh':
            orm.Int(1),
            'Initmag':
            orm.Int(3),
            'ip_mode':
            orm.Str('M'),
            'ip_temp':
            orm.Int(100),
            'ip_mcNstep':
            orm.Int(5000),
            'qm_svec':
            orm.Str('1   -1   0 '),
            'qm_nvec':
            orm.Str('0  0  1'),
            'mode':
            orm.Str('S'),
            'temp':
            orm.Float(50),
            'damping':
            orm.Float(0.01),
            'Nstep':
            orm.Int(10000),
            'timestep':
            orm.Str('1.000e-16'),
            'qpoints':
            orm.Str('D'),
            'plotenergy':
            orm.Int(1),
            'do_avrg':
            orm.Str('Y'),
            'do_sc':
            orm.Str('Q'),
            'do_ams':
            orm.Str('Y'),
            'do_magdos':
            orm.Str('Y'),
            'do_sc_proj':
            orm.Str('Q'),
            'sc_step':
            orm.Int(20),
            'sc_nstep':
            orm.Int(5000),
            'magdos_freq':
            orm.Int(200),
            'magdos_sigma':
            orm.Int(30),
        }
    ),
    'num_machines':
    orm.Int(1),
    'num_mpiprocs_per_machine':
    orm.Int(16),
    'max_wallclock_seconds':
    orm.Int(2000),
    'code':
    orm.Code.get_from_string('uppasd_dev@uppasd_local'),
    'input_filename':
    orm.Str('inpsd.dat'),
    'parser_name':
    orm.Str('uppasd.uppasd_parser'),
    'label':
    orm.Str('uppasd_base_workflow_demo'),
    'description':
    orm.Str('Test base workflow'),
    'prepared_file_folder':
    orm.Str(os.path.join(os.getcwd(), 'AMS_input')),
    'except_filenames':
    orm.List(list=[]),
    'retrieve_list_name':
    orm.List(list=[('*', '.', 0), ('*.json', '.', 0)]),
    'J_model':
    orm.Int(-1),
    'plot_dir':
    orm.Str(current_path),
    'AMSplot':
    orm.Bool('True')
}

job_node = submit(UppASDMagnonSpectraRestartWorkflow, **input_uppasd)

print(f'UppASDAMSPlotWorkflow submitted, PK: {job_node.pk}')
with open('UppASDAMSPlotWorkflow_jobPK.csv', 'w', encoding='utf-8') as f:
    f.write(str(job_node.pk))
