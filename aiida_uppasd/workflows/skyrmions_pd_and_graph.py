#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 13:03:40 2022

@author Qichen Xu
Workchain demo for plot magnon spectra

In this workchain demo we want to use the same method like baserestartworkchain.
This workchain only takes output and do the magnon spectra plot task

For activate J1 J2 J3 .... mode(the nearest model, next nearest mode etc.)
We add one validation function here to check if people want to use this or
offer exchange file directly

several functions are base on codes from UppASD repo like preQ.py and postQ.py
which @Anders bergman
"""

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import vtk

from aiida.engine import ToContext, WorkChain, calcfunction, if_
from aiida.orm import ArrayData, Dict, Int, List, Str

from aiida_uppasd.workflows.base_restart import ASDBaseRestartWorkChain


# Read Location of Atoms
def read_atoms(coord_array_data, max_number_atoms):
    """Transform the atomic positions to VTK structure"""

    points = vtk.vtkPoints()
    number_atoms = 0
    # Read ahead

    # Read all data
    for data in coord_array_data.get_array('coord'):
        if number_atoms <= max_number_atoms:
            coord_x, coord_y, coord_z = float(data[1]), float(data[2]), float(data[3])
            points.InsertPoint(number_atoms, coord_x, coord_y, coord_z)
        number_atoms = number_atoms + 1
    return points, number_atoms


# Read vectors
# We must know the time step and the number of atoms per time
def read_vectors_data(mom_states, number_atoms):
    """Transform the magnetic moments to VTK structure"""
    final_mom_states_x = mom_states.get_array('mom_states_x')[-number_atoms:]
    final_mom_states_y = mom_states.get_array('mom_states_y')[-number_atoms:]
    final_mom_states_z = mom_states.get_array('mom_states_z')[-number_atoms:]
    # Create a Double array which represents the vectors
    vectors = vtk.vtkFloatArray()
    colors = vtk.vtkFloatArray()

    # Define number of elemnts
    vectors.SetNumberOfComponents(3)
    colors.SetNumberOfComponents(1)

    for index in range(number_atoms):
        vec_x, vec_y, vec_z = float(final_mom_states_x[index]), float(final_mom_states_y[index]
                                                                      ), float(final_mom_states_z[index])
        _color = (vec_z + 1.0) / 2.0
        vectors.InsertTuple3(index, vec_x, vec_y, vec_z)
        colors.InsertValue(index, _color)
    return vectors, colors


#----------------------------
# screenshot code begins here
#----------------------------
#A function that takes a renderwindow and saves its contents to a .png file
def screenshot(ren_win, temperature, field, plot_dir):
    """Take a screenshot of the magnetic configuration and save it as a png"""
    win2im = vtk.vtkWindowToImageFilter()
    win2im.ReadFrontBufferOff()
    win2im.SetInput(ren_win)
    #
    povexp = vtk.vtkPOVExporter()
    povexp.SetRenderWindow(ren_win)
    #povexp.SetInput(renWin)
    ren_win.Render()
    # ren_win.SetFileName('realspace_plot_T_{}_B_{}.pov'.format(T,B))
    # povexp.Write()
    #
    to_png = vtk.vtkPNGWriter()
    to_png.SetFileName(f'{plot_dir}/T{temperature}B{field}.png')
    to_png.SetInputConnection(win2im.GetOutputPort())
    to_png.Write()
    return 0


def plot_realspace(points, vectors, colors, temperature, field, plot_dir):  #pylint: disable=too-many-statements,too-many-locals
    """Plot the real space magnetic configuration"""
    ren_win = vtk.vtkRenderWindow()
    ren = vtk.vtkOpenGLRenderer()
    ren_win.AddRenderer(ren)

    # Set color of backgroung
    ren.SetBackground(1.0, 1.0, 1.0)
    ren_win.SetSize(2096, 1280)

    data_test = vtk.vtkPolyData()
    data_scal = vtk.vtkUnstructuredGrid()

    # Read atom positions
    atom_data = points
    data_test.SetPoints(atom_data)
    data_scal.SetPoints(atom_data)

    # Read data for vectors
    vecz, colz = (vectors, colors)
    data_test.GetPointData().SetVectors(vecz)
    data_test.GetPointData().SetScalars(colz)
    data_scal.GetPointData().SetScalars(colz)

    # Create colortable for the coloring of the vectors
    lut = vtk.vtkLookupTable()
    for i in range(0, 128, 1):
        lut.SetTableValue(i, (127.0 - i) / 127.0, i / 127.0, 0, 1)
    for i in range(128, 256, 1):
        lut.SetTableValue(i, 0, (256.0 - i) / 128.0, (i - 128.0) / 128, 1)
    lut.SetTableRange(-1.0, 1.0)
    lut.Build()

    # Set up atoms
    ball = vtk.vtkSphereSource()
    ball.SetRadius(1.00)
    ball.SetThetaResolution(16)
    ball.SetPhiResolution(16)

    balls = vtk.vtkGlyph3DMapper()
    balls.SetInputData(data_test)
    balls.SetSourceConnection(ball.GetOutputPort())
    balls.SetScaleFactor(0.15)
    balls.SetScaleModeToNodata_scaling()
    balls.SetLookupTable(lut)
    balls.Update()

    atom = vtk.vtkLODActor()
    atom.SetMapper(balls)
    atom.GetProperty().SetOpacity(1.5)
    xmin, xmax = atom.GetXRange()
    ymin, ymax = atom.GetYRange()
    zmin, zmax = atom.GetZRange()
    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2
    zmid = (zmin + zmax) / 2
    atom.SetPosition(-xmid, -ymid, -zmid)
    atom.GetProperty().SetSpecular(0.3)
    atom.GetProperty().SetSpecularPower(60)
    atom.GetProperty().SetAmbient(0.2)
    atom.GetProperty().SetDiffuse(0.8)

    # Create vectors
    arrow = vtk.vtkArrowSource()
    arrow.SetTipRadius(0.25)
    arrow.SetShaftRadius(0.15)
    arrow.SetTipResolution(72)
    arrow.SetShaftResolution(72)

    glyph = vtk.vtkGlyph3DMapper()
    glyph.SetSourceConnection(arrow.GetOutputPort())
    glyph.SetInputData(data_test)
    glyph.SetScaleFactor(2.00)
    # Color the vectors according to magnitude
    glyph.SetScaleModeToNodata_scaling()
    glyph.SetLookupTable(lut)
    glyph.SetColorModeToMapScalars()
    glyph.Update()

    vector = vtk.vtkLODActor()
    vector.SetMapper(glyph)
    vector.SetPosition(-xmid, -ymid, -zmid)

    vector.GetProperty().SetSpecular(0.3)
    vector.GetProperty().SetSpecularPower(60)
    vector.GetProperty().SetAmbient(0.2)
    vector.GetProperty().SetDiffuse(0.8)
    vector.GetProperty().SetOpacity(1.0)

    # Scalar bar
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetLookupTable(lut)
    scalar_bar.SetOrientationToHorizontal()
    scalar_bar.SetNumberOfLabels(0)
    scalar_bar.SetPosition(0.1, 0.05)
    scalar_bar.SetWidth(0.85)
    scalar_bar.SetHeight(0.3)
    scalar_bar.GetLabelTextProperty().SetFontSize(8)

    #Depth sorted field
    data_test = vtk.vtkDepthSortPolyData()
    data_test.SetInputData(data_test)
    data_test.SetCamera(ren.GetActiveCamera())
    data_test.SortScalarsOn()
    data_test.Update()

    #cubes
    cube = vtk.vtkCubeSource()
    cubes = vtk.vtkGlyph3DMapper()
    cubes.SetInputConnection(data_test.GetOutputPort())
    cubes.SetSourceConnection(cube.GetOutputPort())
    cubes.SetScaleModeToNodata_scaling()
    cubes.SetScaleFactor(0.995)
    cubes.SetLookupTable(lut)
    cubes.SetColorModeToMapScalars()
    cubes.ScalarVisibilityOn()
    cubes.OrientOff()
    cubes.Update()
    cubes.SetLookupTable(lut)

    cube_actor = vtk.vtkActor()
    cube_actor.SetMapper(cubes)
    cube_actor.GetProperty().SetOpacity(0.05)
    cube_actor.SetPosition(-xmid, -ymid, -zmid)

    #hedgehog
    hhog = vtk.vtkHedgeHog()
    hhog.SetInputData(data_test)
    hhog.SetScaleFactor(5.0)
    hhog_mapper = vtk.vtkPolyDataMapper()
    hhog_mapper.SetInputConnection(hhog.GetOutputPort())
    hhog_mapper.SetLookupTable(lut)
    hhog_mapper.ScalarVisibilityOn()
    hhog_actor = vtk.vtkActor()
    hhog_actor.SetMapper(hhog_mapper)
    hhog_actor.SetPosition(-xmid, -ymid, -zmid)

    #cut plane
    plane = vtk.vtkPlane()
    plane.SetOrigin(data_scal.GetCenter())
    plane.SetNormal(0.0, 0.0, 1.0)
    plane_cut = vtk.vtkCutter()
    plane_cut.SetInputData(data_scal)
    plane_cut.SetInputArrayToProcess(
        0, 0, 0,
        vtk.vtkDataObject().FIELD_ASSOCIATION_POINTS,
        vtk.vtkDataSetAttributes().SCALARS
    )
    plane_cut.SetCutFunction(plane)
    plane_cut.GenerateCutScalarsOff()
    plane_cut.SetSortByToSortByCell()
    clut = vtk.vtkLookupTable()
    clut.SetHueRange(0, .67)
    clut.Build()
    plane_mapper = vtk.vtkPolyDataMapper()
    plane_mapper.SetInputConnection(plane_cut.GetOutputPort())
    plane_mapper.ScalarVisibilityOn()
    plane_mapper.SetScalarRange(data_scal.GetScalarRange())
    plane_mapper.SetLookupTable(clut)
    plane_actor = vtk.vtkActor()
    plane_actor.SetMapper(plane_mapper)

    #clip plane
    plane_clip = vtk.vtkClipDataSet()
    plane_clip.SetInputData(data_scal)
    plane_clip.SetInputArrayToProcess(
        0, 0, 0,
        vtk.vtkDataObject().FIELD_ASSOCIATION_POINTS,
        vtk.vtkDataSetAttributes().SCALARS
    )
    plane_clip.SetClipFunction(plane)
    plane_clip.InsideOutOn()
    clip_mapper = vtk.vtkDataSetMapper()
    clip_mapper.SetInputConnection(plane_cut.GetOutputPort())
    clip_mapper.ScalarVisibilityOn()
    clip_mapper.SetScalarRange(data_scal.GetScalarRange())
    clip_mapper.SetLookupTable(clut)
    clip_actor = vtk.vtkActor()
    clip_actor.SetMapper(clip_mapper)

    # Bounding box
    outline_data = vtk.vtkOutlineFilter()
    outline_data.SetInputData(data_test)
    outline_mapper = vtk.vtkPolyDataMapper()
    outline_mapper.SetInputConnection(outline_data.GetOutputPort())
    outline = vtk.vtkActor()
    outline.SetMapper(outline_mapper)
    outline.GetProperty().SetColor(0, 0, 0)
    outline.GetProperty().SetLineWidth(5.0)
    outline.SetPosition(-xmid, -ymid, -zmid)

    # Text
    txt = vtk.vtkTextActor()
    txt.SetInput('T = TEMP K')
    txtprop = txt.GetTextProperty()
    txtprop.SetFontFamilyToArial()
    txtprop.SetFontSize(36)
    txtprop.SetColor(0, 0, 0)
    txt.SetDisplayPosition(30, 550)

    # Reposition the camera
    ren.GetActiveCamera().Azimuth(0)
    ren.GetActiveCamera().Elevation(0)
    ren.GetActiveCamera().ParallelProjectionOn()
    ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
    ren.GetActiveCamera().SetViewUp(0, 1, 0)
    ren.GetActiveCamera().SetParallelScale(0.55 * ymax)
    _length = max(xmax - xmin, zmax - zmin) / 2
    _height = _length / 0.26795 * 1.1

    ren.GetActiveCamera().SetPosition(0, 0, _height)
    ren.AddActor(vector)

    # Render scene
    iren = vtk.vtkRenderWindowInteractor()
    istyle = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(istyle)
    iren.SetRenderWindow(ren_win)
    iren.Initialize()
    ren_win.Render()
    screenshot(ren_win, temperature, field, plot_dir)


@calcfunction
def store_sknum_data(sky_num_out, sky_avg_num_out):
    """Store the skyrmion number and average skyrmion number as arrays"""
    sk_num_for_plot = ArrayData()
    sk_num_for_plot.set_array('sky_num_out', np.array(sky_num_out.get_list()))
    sk_num_for_plot.set_array('sky_avg_num_out', np.array(sky_avg_num_out.get_list()))
    return sk_num_for_plot


class UppASDSkyrmionsWorkflow(WorkChain):
    """base workchain for skyrmions
    what we have in this workchain:
    1. generate skyrmions phase diagram and single plot via flag sk_plot, \
        generate single plot only if we have one B and one T, \
        otherwise provide phase diagram.
    # 2. generate graph database for machine learning training, here we use \
        DGL https://www.dgl.ai/. if flag 'ML_graph' is 1, we need additional \
        input to generate data. e.g.   !! this target is moved to another workflow
    3. B list
    4. T list
    """

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec.expose_inputs(ASDBaseRestartWorkChain)
        spec.expose_outputs(ASDBaseRestartWorkChain, include=['cumulants'])

        spec.input(
            'sk_plot',
            valid_type=Int,
            help='flag for plotting skyrmions',
            required=False,
        )
        spec.input(
            'sk_number_plot',
            valid_type=Int,
            help='flag for generating graph database ',
            required=False,
        )
        spec.input(
            'average_magnetic_moment_plot',
            valid_type=Int,
            help='flag for average_magnetic_moment plot',
            required=False
        )
        spec.input(
            'average_specific_heat_plot',
            valid_type=Int,
            help='flag for average specific heat plot',
            required=False,
        )
        spec.input(
            'plot_individual',
            valid_type=Int,
            help='flag for plotting skyrmions',
            default=lambda: Int(0),
            required=False
        )
        spec.input(
            'plot_combine',
            valid_type=Int,
            help='flag for plotting skyrmions',
            default=lambda: Int(0),
            required=False,
        )

        spec.input(
            'inpsd_skyr',
            valid_type=Dict,
            help='dict of inpsd.dat',
            required=False,
        )

        spec.input(
            'temperatures',
            valid_type=List,
            help='T list for inpsd.dat',
            required=False,
        )
        spec.input(
            'external_fields',
            valid_type=List,
            help='B list for inpsd.dat',
            required=False,
        )
        spec.input(
            'plot_dir',
            valid_type=Str,
            help='plot dir ',
            required=False,
        )

        spec.output(
            'sk_num_for_plot',
            valid_type=ArrayData,
            help='skyrmions number ',
            required=False,
        )
        spec.outline(
            cls.submits,
            if_(cls.check_for_plot_sk)(
                cls.results_and_plot,
                cls.combined_for_pd,
            ),
            if_(cls.check_for_plot_sk_number)(
                cls.plot_skynumber,
                cls.plot_skynumber_avg,
                cls.plot_skynumber_number_heat_map,
                cls.plot_sk_number_avg_heatmap,
            ),
        )

    def check_for_plot_sk(self):
        """Check if the skyrmion should plotted"""
        try:
            if self.inputs.sk_plot.value > int(0):
                return True
            return False
        except BaseException:
            return False

    def check_for_plot_sk_number(self):
        """Check if the skyrmion number should be plotted"""
        try:
            if self.inputs.sk_number_plot.value > int(0):
                return True
            return False
        except BaseException:
            return False

    def generate_inputs(self):
        """Generate the inputs for the calculation"""
        inputs = self.exposed_inputs(
            ASDBaseRestartWorkChain
        )  #we need take input dict from the BaseRestartWorkchain, not a new one.
        inputs.code = self.inputs.code
        inputs.inpsd_dict = Dict(dict=self.inputs.inpsd_dict)
        inputs.prepared_file_folder = self.inputs.prepared_file_folder
        inputs.except_filenames = self.inputs.except_filenames
        inputs.retrieve_list_name = self.inputs.retrieve_list_name
        return inputs

    def submits(self):
        """Submit the calculations"""
        calculations = {}
        self.inputs.inpsd_dict = {}
        self.inputs.inpsd_dict.update(self.inputs.inpsd_skyr.get_dict())

        for _index_temperature, temperature in enumerate(self.inputs.temperatures):
            self.inputs.inpsd_dict['temp'] = temperature

            self.inputs.inpsd_dict['ip_mcanneal'] = Str(
                f""" 5
                    10000     {float(temperature) + 500}
                    10000     {float(temperature) + 300}
                    10000     {float(temperature) + 100}
                    10000     {float(temperature) + 10}
                    10000     {float(temperature)}"""
            )

            for _index_field, external_field in enumerate(self.inputs.external_fields):
                self.inputs.inpsd_dict['hfield'] = external_field
                self.inputs.inpsd_dict['ip_hfield'] = external_field
                inputs = self.generate_inputs()
                future = self.submit(ASDBaseRestartWorkChain, **inputs)
                self.report(f'Running loop for T with value {temperature} and B with value {external_field}')
                calculations[f'T{_index_temperature}_B{_index_field}'] = future

        return ToContext(**calculations)

    def results_and_plot(self):
        """Collect and plot the results"""
        sky_num_out = []
        sky_avg_num_out = []

        for _index_temperature, _ in enumerate(self.inputs.temperatures):
            sm_list = []
            sam_list = []
            for _index_field, _ in enumerate(self.inputs.external_fields):
                coord_array = self.ctx[f'T{_index_temperature}_B{_index_field}'].get_outgoing(
                ).get_node_by_label('coord')

                # Size of system for plot
                max_number_atoms = 1000000
                mom_states = self.ctx[f'T{_index_temperature}_B{_index_field}'].get_outgoing(
                ).get_node_by_label('mom_states_traj')
                points, number_atoms = read_atoms(coord_array, max_number_atoms)
                vectors, colors = read_vectors_data(mom_states, number_atoms)
                plot_dir = self.inputs.plot_dir.value
                if self.inputs.plot_individual.value > int(0):
                    plot_realspace(points, vectors, colors, _index_temperature, _index_field, plot_dir)

                sk_data = self.ctx[f'T{_index_temperature}_B{_index_field}'].get_outgoing(
                ).get_node_by_label('sk_num_out')
                sm_list.append(sk_data.get_array('sk_number_data')[0])
                sam_list.append(sk_data.get_array('sk_number_data')[1])

            sky_num_out.append(sm_list)
            sky_avg_num_out.append(sam_list)

        sk_num_for_plot = store_sknum_data(List(list=sky_num_out), List(list=sky_avg_num_out))

        self.out('sk_num_for_plot', sk_num_for_plot)

    def combined_for_pd(self):
        """Combine the collected data for the phase diagrame"""
        if self.inputs.plot_individual.value > int(0) and self.inputs.plot_combine.value > int(0):
            skyrmions_singleplot = []
            plot_dir = self.inputs.plot_dir.value
            for _index_temperature, _ in enumerate(self.inputs.temperatures):
                for _index_field, _ in enumerate(self.inputs.external_fields):
                    skyrmions_singleplot.append(
                        Image.open(f'{plot_dir}/T{_index_temperature}B{_index_field}.png').crop((200, 0, 2096, 1280))
                    )  #we need to crop the white edges
            phase_diagram = Image.new(
                'RGB', (648 * len(self.inputs.temperatures), 640 * len(self.inputs.external_fields))
            )
            for _index_temperature, _ in enumerate(self.inputs.temperatures):
                for _index_field, _ in enumerate(self.inputs.external_fields):
                    phase_diagram.paste(
                        skyrmions_singleplot[len(self.inputs.external_fields) * _index_temperature + _index_field],
                        (0 + 648 * _index_temperature, 0 + 640 * _index_field)
                    )
            phase_diagram.save(f'{plot_dir}/PhaseDiagram.png', quality=100)

    def plot_skynumber(self):
        """Plot the skyrmion number"""
        plot_dir = self.inputs.plot_dir.value
        _, axes = plt.subplots()
        for _index_field, external_field in enumerate(self.inputs.external_fields):
            sk_number_list = []
            for _index_temperature, _ in enumerate(self.inputs.temperatures):
                sk_number_list.append(
                    self.ctx[f'T{_index_temperature}_B{_index_field}'].get_outgoing().get_node_by_label('sk_num_out').
                    get_array('sk_number_data')[0]
                )
            axes.plot(self.inputs.temperatures.get_list(), sk_number_list, label=f'B: {external_field[-6:]} T')
        axes.legend(fontsize='xx-small')
        axes.set_ylabel('Skyrmion number')
        axes.set_xlabel('Temperature')
        plt.savefig(f'{plot_dir}/sk_number.png')

    def plot_skynumber_avg(self):
        """Plot the average skyrmion number"""
        plot_dir = self.inputs.plot_dir.value
        _, axes = plt.subplots()
        for _index_field, external_field in enumerate(self.inputs.external_fields):
            sk_number_list = []
            for _index_temperature, _ in enumerate(self.inputs.temperatures):
                sk_number_list.append(
                    self.ctx[f'T{_index_temperature}_B{_index_field}'].get_outgoing().get_node_by_label('sk_num_out').
                    get_array('sk_number_data')[1]
                )
            axes.plot(
                self.inputs.temperatures.get_list(), sk_number_list, label=f'B: {external_field[-6:]} T'
            )  #this should be changed with different B
        axes.legend(fontsize='xx-small')
        axes.set_ylabel('Skyrmion number')
        axes.set_xlabel('Temperature')
        plt.savefig(f'{plot_dir}/sk_number_avg.png')

    def plot_skynumber_number_heat_map(self):
        """Plot the heat map the skyrmion number"""
        plot_dir = self.inputs.plot_dir.value
        heat_map = []
        for _index_field, _ in enumerate(self.inputs.external_fields):
            sk_number_list = []
            for _index_temperature, _ in enumerate(self.inputs.temperatures):
                sk_number_list.append(
                    self.ctx[f'T{_index_temperature}_B{_index_field}'].get_outgoing().get_node_by_label('sk_num_out').
                    get_array('sk_number_data')[0]
                )
            heat_map.append(sk_number_list)
        y_orginal = self.inputs.external_fields.get_list()
        y_for_plot = []
        for i in y_orginal:
            y_for_plot.append(float(i.split()[-1]))

        fig = go.Figure(
            data=go.Heatmap(
                z=heat_map,
                x=self.inputs.temperatures.get_list(),
                y=y_for_plot,
                zsmooth='best',
                zmid=0,
            )
        )

        fig.update_xaxes(range=[self.inputs.temperatures.get_list()[0], self.inputs.temperatures.get_list()[-1]])
        fig.update_yaxes(range=[y_for_plot[0], y_for_plot[-1]])
        fig.update_traces(colorscale='Jet', selector=dict(type='heatmap'))
        fig.write_image(f'{plot_dir}/sk_number_heatmap.png')

    def plot_sk_number_avg_heatmap(self):
        """Plot the heat map of the average skyrmion number"""
        plot_dir = self.inputs.plot_dir.value
        heat_map = []
        for _index_field, _ in enumerate(self.inputs.external_fields):
            sk_number_list = []
            for _index_temperature, _ in enumerate(self.inputs.temperatures):
                sk_number_list.append(
                    self.ctx[f'T{_index_temperature}_B{_index_field}'].get_outgoing().get_node_by_label('sk_num_out').
                    get_array('sk_number_data')[1]
                )
            heat_map.append(sk_number_list)
        y_orginal = self.inputs.external_fields.get_list()
        y_for_plot = []
        for i in y_orginal:
            y_for_plot.append(float(i.split()[-1]))

        fig = go.Figure(
            data=go.Heatmap(
                z=heat_map,
                x=self.inputs.temperatures.get_list(),
                y=y_for_plot,
                zsmooth='best',
                zmid=0,
            )
        )

        fig.update_xaxes(range=[self.inputs.temperatures.get_list()[0], self.inputs.temperatures.get_list()[-1]])
        fig.update_yaxes(range=[y_for_plot[0], y_for_plot[-1]])
        fig.update_traces(colorscale='Jet', selector={'type': 'heatmap'})
        fig.write_image(f'{plot_dir}/sk_number_heatmap_avg.png')
