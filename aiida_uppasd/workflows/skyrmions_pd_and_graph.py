#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 13:03:40 2022

@author Qichen Xu
Workchain demo for plot magnon spectra

In this workchain demo we want to use the same method like baserestartworkchain.
This workchain only takes output and do the magnon spectra plot task

For activate J1 J2 J3 .... mode(the nearest model, next nearest mode etc.) We add one validation function here to check if people want to use this or offer exchange file directly

several functions are base on codes from UppASD repo like preQ.py and postQ.py which @Anders bergman
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
def readAtoms(coord_array_data, Nmax):

    points = vtk.vtkPoints()
    nrAtoms = 0
    # Read ahead

    # Read all data
    for data in coord_array_data.get_array('coord'):
        if nrAtoms <= Nmax:
            x, y, z = float(data[1]), float(data[2]), float(data[3])
            #print "a ", a, " x ", x, " y ", y, " z ", z
            #points.InsertPoint(a, x, y, z)
            points.InsertPoint(nrAtoms, x, y, z)
        nrAtoms = nrAtoms + 1
    return points, nrAtoms


# Read vectors
# We must know the time step and the number of atoms per time
def readVectorsData(mom_states, nrAtoms, Nmax):
    final_mom_states_x = mom_states.get_array('mom_states_x')[-nrAtoms:]
    final_mom_states_y = mom_states.get_array('mom_states_y')[-nrAtoms:]
    final_mom_states_z = mom_states.get_array('mom_states_z')[-nrAtoms:]
    # Create a Double array which represents the vectors
    vectors = vtk.vtkFloatArray()
    colors = vtk.vtkFloatArray()

    # Define number of elemnts
    vectors.SetNumberOfComponents(3)
    colors.SetNumberOfComponents(1)

    for i in range(nrAtoms):
        x, y, z = float(final_mom_states_x[i]), float(final_mom_states_y[i]), float(final_mom_states_z[i])
        m = (z + 1.0) / 2.0
        vectors.InsertTuple3(i, x, y, z)
        colors.InsertValue(i, m)
        i = i + 1
    return vectors, colors


#----------------------------
# Screenshot code begins here
#----------------------------
#A function that takes a renderwindow and saves its contents to a .png file
def Screenshot(renWin, T, B, plot_dir):
    global number_of_screenshots
    win2im = vtk.vtkWindowToImageFilter()
    win2im.ReadFrontBufferOff()
    win2im.SetInput(renWin)
    #
    povexp = vtk.vtkPOVExporter()
    povexp.SetRenderWindow(renWin)
    #povexp.SetInput(renWin)
    renWin.Render()
    # povexp.SetFileName('realspace_plot_T_{}_B_{}.pov'.format(T,B))
    # povexp.Write()
    #
    toPNG = vtk.vtkPNGWriter()
    toPNG.SetFileName(f'{plot_dir}/T{T}B{B}.png')
    toPNG.SetInputConnection(win2im.GetOutputPort())
    toPNG.Write()
    return 0


def plot_realspace(points, nrAtoms, vectors, colors, Nmax, T, B, plot_dir):
    renWin = vtk.vtkRenderWindow()
    ren = vtk.vtkOpenGLRenderer()
    renWin.AddRenderer(ren)

    # Set color of backgroung
    ren.SetBackground(1.0, 1.0, 1.0)
    renWin.SetSize(2096, 1280)

    Datatest = vtk.vtkPolyData()
    DataScal = vtk.vtkUnstructuredGrid()

    # Read atom positions
    atomData = points
    Datatest.SetPoints(atomData)
    DataScal.SetPoints(atomData)

    # Read data for vectors
    vecz, colz = (vectors, colors)
    Datatest.GetPointData().SetVectors(vecz)
    Datatest.GetPointData().SetScalars(colz)
    DataScal.GetPointData().SetScalars(colz)

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
    balls.SetInputData(Datatest)
    balls.SetSourceConnection(ball.GetOutputPort())
    balls.SetScaleFactor(0.15)
    balls.SetScaleModeToNoDataScaling()
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
    glyph.SetInputData(Datatest)
    glyph.SetScaleFactor(2.00)
    # Color the vectors according to magnitude
    glyph.SetScaleModeToNoDataScaling()
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
    scalarBar = vtk.vtkScalarBarActor()
    scalarBar.SetLookupTable(lut)
    scalarBar.SetOrientationToHorizontal()
    scalarBar.SetNumberOfLabels(0)
    scalarBar.SetPosition(0.1, 0.05)
    scalarBar.SetWidth(0.85)
    scalarBar.SetHeight(0.3)
    scalarBar.GetLabelTextProperty().SetFontSize(8)

    #Depth sorted field
    dsDatatest = vtk.vtkDepthSortPolyData()
    dsDatatest.SetInputData(Datatest)
    dsDatatest.SetCamera(ren.GetActiveCamera())
    dsDatatest.SortScalarsOn()
    dsDatatest.Update()

    #cubes
    cube = vtk.vtkCubeSource()
    cubes = vtk.vtkGlyph3DMapper()
    cubes.SetInputConnection(dsDatatest.GetOutputPort())
    cubes.SetSourceConnection(cube.GetOutputPort())
    cubes.SetScaleModeToNoDataScaling()
    cubes.SetScaleFactor(0.995)
    cubes.SetLookupTable(lut)
    cubes.SetColorModeToMapScalars()
    cubes.ScalarVisibilityOn()
    cubes.OrientOff()
    cubes.Update()
    cubes.SetLookupTable(lut)

    cubeActor = vtk.vtkActor()
    cubeActor.SetMapper(cubes)
    cubeActor.GetProperty().SetOpacity(0.05)
    cubeActor.SetPosition(-xmid, -ymid, -zmid)

    #hedgehog
    hhog = vtk.vtkHedgeHog()
    hhog.SetInputData(Datatest)
    hhog.SetScaleFactor(5.0)
    hhogMapper = vtk.vtkPolyDataMapper()
    hhogMapper.SetInputConnection(hhog.GetOutputPort())
    hhogMapper.SetLookupTable(lut)
    hhogMapper.ScalarVisibilityOn()
    hhogActor = vtk.vtkActor()
    hhogActor.SetMapper(hhogMapper)
    hhogActor.SetPosition(-xmid, -ymid, -zmid)

    #cut plane
    plane = vtk.vtkPlane()
    plane.SetOrigin(DataScal.GetCenter())
    plane.SetNormal(0.0, 0.0, 1.0)
    planeCut = vtk.vtkCutter()
    planeCut.SetInputData(DataScal)
    planeCut.SetInputArrayToProcess(
        0, 0, 0,
        vtk.vtkDataObject().FIELD_ASSOCIATION_POINTS,
        vtk.vtkDataSetAttributes().SCALARS
    )
    planeCut.SetCutFunction(plane)
    planeCut.GenerateCutScalarsOff()
    planeCut.SetSortByToSortByCell()
    clut = vtk.vtkLookupTable()
    clut.SetHueRange(0, .67)
    clut.Build()
    planeMapper = vtk.vtkPolyDataMapper()
    planeMapper.SetInputConnection(planeCut.GetOutputPort())
    planeMapper.ScalarVisibilityOn()
    planeMapper.SetScalarRange(DataScal.GetScalarRange())
    planeMapper.SetLookupTable(clut)
    planeActor = vtk.vtkActor()
    planeActor.SetMapper(planeMapper)

    #clip plane
    planeClip = vtk.vtkClipDataSet()
    planeClip.SetInputData(DataScal)
    planeClip.SetInputArrayToProcess(
        0, 0, 0,
        vtk.vtkDataObject().FIELD_ASSOCIATION_POINTS,
        vtk.vtkDataSetAttributes().SCALARS
    )
    planeClip.SetClipFunction(plane)
    planeClip.InsideOutOn()
    clipMapper = vtk.vtkDataSetMapper()
    clipMapper.SetInputConnection(planeCut.GetOutputPort())
    clipMapper.ScalarVisibilityOn()
    clipMapper.SetScalarRange(DataScal.GetScalarRange())
    clipMapper.SetLookupTable(clut)
    clipActor = vtk.vtkActor()
    clipActor.SetMapper(clipMapper)

    # Bounding box
    outlineData = vtk.vtkOutlineFilter()
    outlineData.SetInputData(Datatest)
    outlineMapper = vtk.vtkPolyDataMapper()
    outlineMapper.SetInputConnection(outlineData.GetOutputPort())
    outline = vtk.vtkActor()
    outline.SetMapper(outlineMapper)
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
    d = ren.GetActiveCamera().GetDistance()
    ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
    ren.GetActiveCamera().SetViewUp(0, 1, 0)
    ren.GetActiveCamera().SetParallelScale(0.55 * ymax)
    l = max(xmax - xmin, zmax - zmin) / 2
    h = l / 0.26795 * 1.1

    ren.GetActiveCamera().SetPosition(0, 0, h)
    ren.AddActor(vector)

    # Render scene
    iren = vtk.vtkRenderWindowInteractor()
    istyle = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(istyle)
    iren.SetRenderWindow(renWin)
    iren.Initialize()
    renWin.Render()
    Screenshot(renWin, T, B, plot_dir)


@calcfunction
def store_sknum_data(sky_num_out, sky_avg_num_out):
    sk_num_for_plot = ArrayData()
    sk_num_for_plot.set_array('sky_num_out', np.array(sky_num_out.get_list()))
    sk_num_for_plot.set_array('sky_avg_num_out', np.array(sky_avg_num_out.get_list()))
    return sk_num_for_plot


class UppASDSkyrmionsWorkflow(WorkChain):
    """base workchain for skyrmions
    what we have in this workchain:
    1. generate skyrmions phase diagram and single plot via flag sk_plot, generate single plot only if we have one B and one T, otherwise provide phase diagram.
    # 2. generate graph database for machine learning training, here we use DGL https://www.dgl.ai/. if flag 'ML_graph' is 1, we need additional input to generate data. e.g.   !! this target is moved to another workflow
    3. B list
    4. T list
    """

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec.expose_inputs(ASDBaseRestartWorkChain)
        spec.expose_outputs(ASDBaseRestartWorkChain, include=['cumulants'])

        spec.input('sk_plot', valid_type=Int, help='flag for plotting skyrmions', required=False)
        spec.input('sk_number_plot', valid_type=Int, help='flag for generating graph database ', required=False)
        spec.input(
            'average_magnetic_moment_plot',
            valid_type=Int,
            help='flag for average_magnetic_moment plot',
            required=False
        )
        spec.input(
            'average_specific_heat_plot', valid_type=Int, help='flag for average specific heat plot', required=False
        )
        spec.input(
            'plot_individual',
            valid_type=Int,
            help='flag for plotting skyrmions',
            default=lambda: Int(0),
            required=False
        )
        spec.input(
            'plot_combine', valid_type=Int, help='flag for plotting skyrmions', default=lambda: Int(0), required=False
        )

        spec.input(
            'inpsd_skyr', valid_type=Dict, help='dict of inpsd.dat', required=False
        )  # default=lambda: Dict(dict={})

        spec.input(
            'temperatures', valid_type=List, help='T list for inpsd.dat', required=False
        )  # default=lambda: Dict(dict={})
        spec.input(
            'external_fields', valid_type=List, help='B list for inpsd.dat', required=False
        )  # default=lambda: Dict(dict={})
        spec.input('plot_dir', valid_type=Str, help='plot dir ', required=False)

        spec.output('sk_num_for_plot', valid_type=ArrayData, help='skyrmions number ', required=False)
        spec.outline(
            cls.submits,
            if_(cls.check_for_plot_sk)(
                cls.results_and_plot,
                cls.combined_for_PD,
            ),
            if_(cls.check_for_plot_sk_number)(
                cls.plot_skynumber,
                cls.plot_skynumber_avg,
                cls.plot_skynumber_number_heat_map,
                cls.plot_skynumber_number_avg_heat_map,
            ),
        )

    def check_for_plot_sk(self):
        try:
            if self.inputs.sk_plot.value > int(0):
                return True
            return False
        except BaseException:
            return False

    def check_for_plot_sk_number(self):
        try:
            if self.inputs.sk_number_plot.value > int(0):
                return True
            return False
        except BaseException:
            return False

    def generate_inputs(self):
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
        calculations = {}
        self.inputs.inpsd_dict = {}
        self.inputs.inpsd_dict.update(self.inputs.inpsd_skyr.get_dict())

        for idx_T, temperature in enumerate(self.inputs.temperatures):
            self.inputs.inpsd_dict['temp'] = temperature
            #self.inputs.inpsd_dict['ip_temp'] = temperature

            self.inputs.inpsd_dict['ip_mcanneal'] = Str(
                f""" 5
                    10000     {float(temperature) + 500}
                    10000     {float(temperature) + 300}
                    10000     {float(temperature) + 100}
                    10000     {float(temperature) + 10}
                    10000     {float(temperature)}"""
            )

            for idx_B, externel_B in enumerate(self.inputs.external_fields):
                self.inputs.inpsd_dict['hfield'] = externel_B
                self.inputs.inpsd_dict['ip_hfield'] = externel_B
                inputs = self.generate_inputs()
                future = self.submit(ASDBaseRestartWorkChain, **inputs)
                self.report(f'Running loop for T with value {temperature} and B with value {externel_B}')
                calculations['T' + str(idx_T) + '_' + 'B' + str(idx_B)] = future

        return ToContext(**calculations)

    def results_and_plot(self):
        sky_num_out = []
        sky_avg_num_out = []

        for idx_T, temperature in enumerate(self.inputs.temperatures):
            sm_list = []
            sam_list = []
            for idx_B, externel_B in enumerate(self.inputs.external_fields):
                coord_array = self.ctx['T' + str(idx_T) + '_' + 'B' +
                                       str(idx_B)].get_outgoing().get_node_by_label('coord')

                # Size of system for plot
                Nmax = 1000000
                mom_states = self.ctx['T' + str(idx_T) + '_' + 'B' +
                                      str(idx_B)].get_outgoing().get_node_by_label('mom_states_traj')
                points, nrAtoms = readAtoms(coord_array, Nmax)
                vectors, colors = readVectorsData(mom_states, nrAtoms, Nmax)
                plot_dir = self.inputs.plot_dir.value
                if self.inputs.plot_individual.value > int(0):
                    plot_realspace(points, nrAtoms, vectors, colors, Nmax, idx_T, idx_B, plot_dir)

                sk_data = self.ctx['T' + str(idx_T) + '_' + 'B' +
                                   str(idx_B)].get_outgoing().get_node_by_label('sk_num_out')
                sm_list.append(sk_data.get_array('sk_number_data')[0])
                sam_list.append(sk_data.get_array('sk_number_data')[1])

            sky_num_out.append(sm_list)
            sky_avg_num_out.append(sam_list)

        sk_num_for_plot = store_sknum_data(List(list=sky_num_out), List(list=sky_avg_num_out))

        self.out('sk_num_for_plot', sk_num_for_plot)

    def combined_for_PD(self):
        if self.inputs.plot_individual.value > int(0) and self.inputs.plot_combine.value > int(0):
            skyrmions_singleplot = []
            plot_dir = self.inputs.plot_dir.value
            for idx_T, temperature in enumerate(self.inputs.temperatures):
                for idx_B, externel_B in enumerate(self.inputs.external_fields):
                    skyrmions_singleplot.append(
                        Image.open(f'{plot_dir}/T{idx_T}B{idx_B}.png').crop((200, 0, 2096, 1280))
                    )  #we need to crop the white edges
            phase_diagram = Image.new(
                'RGB', (648 * len(self.inputs.temperatures), 640 * len(self.inputs.external_fields))
            )
            for idx_T, temperature in enumerate(self.inputs.temperatures):
                for idx_B, externel_B in enumerate(self.inputs.external_fields):
                    phase_diagram.paste(
                        skyrmions_singleplot[len(self.inputs.external_fields) * idx_T + idx_B],
                        (0 + 648 * idx_T, 0 + 640 * idx_B)
                    )
            phase_diagram.save(f'{plot_dir}/PhaseDiagram.png', quality=100)

    def plot_skynumber(self):
        plot_dir = self.inputs.plot_dir.value
        _, ax = plt.subplots()
        for idx_B, externel_B in enumerate(self.inputs.external_fields):
            sk_number_list = []
            for idx_T, _ in enumerate(self.inputs.temperatures):
                sk_number_list.append(
                    self.ctx['T' + str(idx_T) + '_' + 'B' +
                             str(idx_B)].get_outgoing().get_node_by_label('sk_num_out').get_array('sk_number_data')[0]
                )
            ax.plot(self.inputs.temperatures.get_list(), sk_number_list, label=f'B: {externel_B[-6:]} T')
        ax.legend(fontsize='xx-small')
        ax.set_ylabel('Skyrmion number')
        ax.set_xlabel('Temperature')
        plt.savefig(f'{plot_dir}/sk_number.png')

    def plot_skynumber_avg(self):
        plot_dir = self.inputs.plot_dir.value
        _, ax = plt.subplots()
        for idx_B, externel_B in enumerate(self.inputs.external_fields):
            sk_number_list = []
            for idx_T, _ in enumerate(self.inputs.temperatures):
                sk_number_list.append(
                    self.ctx['T' + str(idx_T) + '_' + 'B' +
                             str(idx_B)].get_outgoing().get_node_by_label('sk_num_out').get_array('sk_number_data')[1]
                )
            ax.plot(
                self.inputs.temperatures.get_list(), sk_number_list, label=f'B: {externel_B[-6:]} T'
            )  #this should be changed with different B
        ax.legend(fontsize='xx-small')
        ax.set_ylabel('Skyrmion number')
        ax.set_xlabel('Temperature')
        plt.savefig(f'{plot_dir}/sk_number_avg.png')

    def plot_skynumber_number_heat_map(self):
        plot_dir = self.inputs.plot_dir.value
        heat_map = []
        for idx_B, externel_B in enumerate(self.inputs.external_fields):
            sk_number_list = []
            for idx_T, temperature in enumerate(self.inputs.temperatures):
                sk_number_list.append(
                    self.ctx['T' + str(idx_T) + '_' + 'B' +
                             str(idx_B)].get_outgoing().get_node_by_label('sk_num_out').get_array('sk_number_data')[0]
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

    def plot_skynumber_number_avg_heat_map(self):
        plot_dir = self.inputs.plot_dir.value
        heat_map = []
        for idx_B, externel_B in enumerate(self.inputs.external_fields):
            sk_number_list = []
            for idx_T, temperature in enumerate(self.inputs.temperatures):
                sk_number_list.append(
                    self.ctx['T' + str(idx_T) + '_' + 'B' +
                             str(idx_B)].get_outgoing().get_node_by_label('sk_num_out').get_array('sk_number_data')[1]
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
        fig.write_image(f'{plot_dir}/sk_number_heatmap_avg.png')
