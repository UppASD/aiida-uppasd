"""Set of functions to plot uppasd results"""
import os
from typing import Union

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy import ndimage
import vtk

from aiida import orm


def plot_skyrmion_number(node: orm.Node, plot_dir: Union[str, os.PathLike], method: str = 'plain'):
    """Plot the skyrmion number"""

    if method.lower() in ['plain', 'heatmap_plain']:
        _axis = 0
    if method.lower() == ['average', 'heatmap_average']:
        _axis = 1

    if method.lower() not in ['plain', 'average', 'heatmap_plain', 'heatmap_average']:
        raise ValueError('The method should be either "plain" or "average"')

    _sk_data = []
    _bfield = []
    _temperature = []
    for _sub_node in node.called_descendants:
        _sk_data.append(
            _sub_node.get_outgoing().get_node_by_label('sk_num_out').get_array('sk_number_data')[0][:, _axis]
        )
        _bfield.append(_sub_node.inputs.inpsd_dict['hfield'].split()[-1])
        _temperature.append(_sub_node.inputs.inpsd_dict['temp'])

    if method.lower() in ['plain', 'average']:
        _bfield_unique = np.unique(_bfield)[0]

        fig = go.Figure()

        for _field in _bfield_unique:
            _temperature_curr = np.asarray(_temperature)[np.where(np.asarray(_bfield) == _field)[0]]
            _sk_number_curr = np.asarray(_sk_data[np.where(np.asarray(_bfield) == _field)[0]])

            fig.add_trace(go.Scatter(_temperature_curr, _sk_number_curr, name=f'B: {_field} T'))
        fig.update_layout(
            xaxis_title='Temperature',
            yaxis_title='Skyrmion number',
        )

        fig.write_image(f'{plot_dir}/sk_number_{method}.png')

    if method.lower() in ['heatmap_plain', 'heatmap_average']:

        fig = go.Figure(data=go.Heatmap(
            z=_sk_data,
            x=_temperature,
            y=_bfield,
            zsmooth='best',
            zmid=0,
        ))

        fig.update_xaxes(range=[min(_temperature), max(_temperature)])
        fig.update_yaxes(range=[min(_bfield), max(_bfield)])
        fig.update_traces(colorscale='Jet', selector=dict(type='heatmap'))
        fig.write_image(f'{plot_dir}/sk_number_heatmap.png')


def combined_for_pd(node: orm.Node, plot_dir: Union[str, os.PathLike]):
    """Combine the collected data for the phase diagram"""
    if node.inputs.plot_individual.value > int(0) and node.inputs.plot_combine.value > int(0):
        skyrmions_singleplot = []
        for _index_temperature, _ in enumerate(node.inputs.temperatures):
            for _index_field, _ in enumerate(node.inputs.external_fields):
                skyrmions_singleplot.append(
                    Image.open(f'{plot_dir}/T{_index_temperature}B{_index_field}.png').crop((200, 0, 2096, 1280))
                )  #we need to crop the white edges
        phase_diagram = Image.new('RGB', (648 * len(node.inputs.temperatures), 640 * len(node.inputs.external_fields)))
        for _index_temperature, _ in enumerate(node.inputs.temperatures):
            for _index_field, _ in enumerate(node.inputs.external_fields):
                phase_diagram.paste(
                    skyrmions_singleplot[len(node.inputs.external_fields) * _index_temperature + _index_field],
                    (0 + 648 * _index_temperature, 0 + 640 * _index_field)
                )
        phase_diagram.save(f'{plot_dir}/PhaseDiagram.png', quality=100)


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


def plot_configuration(node: orm.Node, plot_dir: Union[str, os.PathLike]):
    """Collect and plot the results"""

    for _sub_node in node.called_descendants:

        _index_temperature = node.inputs.temperatures.get_list().index(_sub_node.inputs.inpsd_dict.get_dict()['temp'])
        _index_field = node.inputs.external_fields.get_list().index(_sub_node.inputs.inpsd_dict.get_dict()['hfield'])

        coord_array = _sub_node.get_node_by_label('coord')

        # Size of system for plot
        max_number_atoms = 1000000
        mom_states = _sub_node.get_outgoing().get_node_by_label('mom_states_traj')
        points, number_atoms = read_atoms(coord_array, max_number_atoms)
        vectors, colors = read_vectors_data(mom_states, number_atoms)
        if node.inputs.plot_individual.value > int(0):
            plot_realspace(points, vectors, colors, _index_temperature, _index_field, plot_dir)


def plot_result(node: orm.Node, plot_dir: Union[str, os.PathLike], curie_temperature: Union[float, int, None] = None):
    """
    Basic plot function
    Note that we use normalized axis like T/Tc in all figures
    """

    fig_mag = go.Figure()
    fig_ene = go.Figure()

    _norm_factor = curie_temperature if curie_temperature is not None else 1

    for _sub_node in node.called_descendants:
        data = _sub_node.outputs.temperature_output.get_dict()
        fig_mag.add_trace(
            go.Scatter(
                np.array(data['temperature']) / _norm_factor,
                np.array(data['magnetization']) / np.array(data['magnetization'][0]),
                name=f'N={_sub_node.inputs.inpsd_dict.get_dict()["necll"].split()[0]}',
            )
        )
        fig_ene.add_trace(
            go.Scatter(
                np.array(data['temperature']) / _norm_factor,
                np.array(data['energy']),
                name=f'N={_sub_node.inputs.inpsd_dict.get_dict()["necll"].split()[0]}',
            )
        )

    fig_mag.update_layout(
        xaxis_title='T/T<sub>c</sub>',
        yaxis_title='M/M<sub>sat</sub>',
    )
    fig_ene.update_layout(
        xaxis_title='T/T<sub>c</sub>',
        yaxis_title='Energy (mRy)',
    )

    fig_mag.write_image(f'{plot_dir}/temperature-magnetization.png')
    fig_ene.write_image(f'{plot_dir}/temperature-energy.png')


def plot_ams(node: orm.Node, plot_dir: Union[str, os.PathLike]):  # pylint: disable=too-many-locals
    """Process results and basic plot"""

    ams_plot_var_out = node.outputs.ams_plot_var.get_dict()
    timestep = ams_plot_var_out['timestep']
    sc_step = ams_plot_var_out['sc_step']
    sqw_x = ams_plot_var_out['sqw_x']
    sqw_y = ams_plot_var_out['sqw_y']
    ams_t = ams_plot_var_out['ams']
    axidx_abs = ams_plot_var_out['axidx_abs']
    ams_dist_col = ams_plot_var_out['ams_dist_col']
    axlab = ams_plot_var_out['axlab']
    _ = plt.figure(figsize=[8, 5])
    axes = plt.subplot(111)
    hbar = float(4.135667662e-15)
    emax = 0.5 * float(hbar) / (float(timestep) * float(sc_step)) * 1e3
    #print('emax_sqw=',emax)
    sqw_x = np.array(sqw_x)
    sqw_y = np.array(sqw_y)
    #ams = np.array(ams)
    sqw_temp = (sqw_x**2.0 + sqw_y**2.0)**0.5
    sqw_temp[:, 0] = sqw_temp[:, 0] / 100.0
    sqw_temp = sqw_temp.T / sqw_temp.T.max(axis=0)
    sqw_temp = ndimage.gaussian_filter1d(
        sqw_temp,
        sigma=1,
        axis=1,
        mode='constant',
    )
    sqw_temp = ndimage.gaussian_filter1d(
        sqw_temp,
        sigma=5,
        axis=0,
        mode='reflect',
    )
    plt.imshow(
        sqw_temp,
        cmap=plt.get_cmap('jet'),
        interpolation='antialiased',
        origin='lower',
        extent=[axidx_abs[0], axidx_abs[-1], 0, emax],
    )  # Note here, we could choose color bar here.
    ams_t = np.array(ams_t)
    plt.plot(
        ams_t[:, 0] / ams_t[-1, 0] * axidx_abs[-1],
        ams_t[:, 1:ams_dist_col],
        'white',
        lw=1,
    )
    plt.colorbar()
    plt.xticks(axidx_abs, axlab)
    plt.xlabel('q')
    plt.ylabel('Energy (meV)')
    plt.autoscale(tight=False)
    axes.set_aspect('auto')
    plt.grid(b=True, which='major', axis='x')
    plt.savefig(f'{plot_dir}/AMS.png')


def plot_average_specific_heat(node: orm.Node, plot_dir: Union[str, os.PathLike]):
    """Plot the average specific heat"""

    #plot the line for one B

    _energy = []
    _bfield = []
    _temperature = []
    for _sub_node in node.called_descendants:
        _energy.append(_sub_node.outputs.cumulats.get_dict()['energy'])
        _temperature.append(_sub_node.inputs.inpsd_dict.get_dict()['temp'])
        _bfield.append(_sub_node.inputs.inpsd_dict.get_dict()['hfield'])

    _bfield_unique = np.unique(_bfield)[0]

    fig = go.Figure()

    heat_map = []

    for _field in _bfield_unique:
        _temperature_curr = np.asarray(_temperature)[np.where(np.asarray(_bfield) == _field)[0]]
        _energy_curr = np.asarray(_energy[np.where(np.asarray(_bfield) == _field)[0]])

        de_dt = np.gradient(np.array(_energy_curr)) / np.gradient(np.array(_temperature_curr))
        heat_map.append(de_dt.tolist())
        de_dt = (de_dt / de_dt[0]).tolist()

        fig.add_trace(go.Scatter(_temperature_curr, de_dt, name=f'B={_field}'))
    fig.update_layout(xaxis_title='Temperature', yaxis_title='C<sub>v</sub>')
    fig.write_image(f'{plot_dir}/CV_T.png')
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heat_map,
        x=_temperature,
        y=_bfield,
        zsmooth='best',
    ))

    fig_heatmap.update_xaxes(range=[min(_temperature), max(_temperature)])
    fig_heatmap.update_yaxes(range=[int(min(_bfield)), int(max(_bfield))])
    fig_heatmap.update_traces(colorscale='Jet', selector=dict(type='heatmap'))
    fig_heatmap.write_image(f'{plot_dir}/average_specific_heat.png')


def plot_average_magnetic_moment(node: orm.Node, plot_dir: Union[str, os.PathLike]):
    """Plot the average magnetic moment"""
    heat_map = []
    _temperature = []
    _bfield = []
    for _sub_node in node.called_descendats:
        _temperature.append(_sub_node.inputs.inpsd_dict.get_dict()['temp'])
        _bfield.append(_sub_node.inputs.inpsd_dcit.get_dict()['hfield'])
        heat_map.append(_sub_node.outputs.cumulants.get_dict()['magnetization'])

    fig = go.Figure(data=go.Heatmap(
        z=heat_map,
        x=_temperature,
        y=_bfield,
        zsmooth='best',
    ))
    fig.update_xaxes(range=[min(_temperature), max(_temperature)])
    fig.update_yaxes(range=[int(min(_bfield)), int(max(_bfield))])
    fig.update_traces(colorscale='Jet', selector=dict(type='heatmap'))
    fig.write_image(f'{plot_dir}/average_magnetic_moment.png')


def plot_pd(
    plot_dir,
    heat_map,
    x_label_list,
    y_label_list,
    plot_name,
    xlabel,
    ylabel,
):
    #pylint: disable=too-many-arguments
    """Plotting the phase diagram"""
    fig = go.Figure(data=go.Heatmap(
        z=heat_map,
        x=x_label_list,
        y=y_label_list,
        zsmooth='best',
    ))
    fig.update_xaxes(range=[int(float(x_label_list[0])), int(float(x_label_list[-1]))])
    fig.update_yaxes(range=[int(float(y_label_list[0])), int(float(y_label_list[-1]))])
    fig.update_traces(colorscale='Jet', selector=dict(type='heatmap'))
    fig.update_layout(yaxis={'title': f'{xlabel}', 'tickangle': -90}, xaxis={'title': f'{ylabel}'})
    fig.write_image(f"{plot_dir}/{plot_name.replace(' ', '_')}.png")


def plot_line(
    x_data,
    y_data,
    line_name_list,
    x_label,
    y_label,
    plot_path,
    plot_name,
    leg_name_list,
):
    #pylint: disable=too-many-arguments
    """Line plot"""
    #Since we need all lines in one plot here x and y should be a dict with the line name on that
    plt.figure()
    _, axes = plt.subplots()
    for index, _entry in enumerate(line_name_list):
        axes.plot(
            x_data,
            y_data[_entry],
            label=f'{leg_name_list[index]}',
        )
    axes.legend()
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    plt.savefig(f'{plot_path}/{plot_name}.png')
    plt.close()


def plot_mag_phase_diagram(node: orm.Node):
    """Plot the magnetization phase diagram"""
    y_label_list = []
    for i in node.inputs.external_fields.get_list():
        y_label_list.append(float(max(np.array(i.split()))))

    for _, cell_size in enumerate(node.inputs.cell_size):
        pd_dict = node.outputs.thermal_dynamic_output.get_dict()[f'{cell_size}']
        pd_for_plot = []
        for i in pd_dict.keys():
            pd_for_plot.append(pd_dict[i]['magnetization'])
        plot_pd(
            node.inputs.plot_dir.value, pd_for_plot, node.inputs.temperatures.get_list(), y_label_list,
            ('M_T' + str(cell_size)), 'B', 'T'
        )


def plot_mag_temp(node: orm.Node):
    """Plot the magnetization as a function of temperature"""
    line_name_list = []
    leg_name_list = []
    line_for_plot = {}
    for _, cell_size in enumerate(node.inputs.cell_size):
        line_dict = node.outputs.thermal_dynamic_output.get_dict()[f'{cell_size}']
        leg_name_list.append(('Cell:' + cell_size.replace(' ', '_')))
        index = 0
        for i in line_dict.keys():
            line_for_plot[f"{cell_size.replace(' ', '_')}_{index}"] = line_dict[i]['magnetization']
            line_name_list.append(f"{cell_size.replace(' ', '_')}_{index}")
            index = index + 1
    plot_line(
        node.inputs.temperatures.get_list(),
        line_for_plot,
        line_name_list,
        'T',
        'M',
        node.inputs.plot_dir.value,
        'M_T',
        leg_name_list,
    )


def plot_cev_phase_diagram(node: orm.Node):
    """Check if the specific heat phase diagram should be produced"""
    y_label_list = []
    for i in node.inputs.external_fields.get_list():
        y_label_list.append(float(max(np.array(i.split()))))

    for _, cell_size in enumerate(node.inputs.cell_size):
        pd_dict = node.outputs.thermal_dynamic_output.get_dict()[f'{cell_size}']
        pd_for_plot = []
        for i in pd_dict.keys():
            pd_for_plot.append(pd_dict[i]['specific_heat'])
        plot_pd(
            node.inputs.plot_dir.value, pd_for_plot, node.inputs.temperatures.get_list(), y_label_list,
            ('specific_heat_T' + str(cell_size)), 'B', 'T'
        )


def plot_specific_heat_temp(node: orm.Node):
    """Plot the specific head as as function of temperature"""
    line_name_list = []
    leg_name_list = []
    line_for_plot = {}
    for _, cell_size in enumerate(node.inputs.cell_size):
        line_dict = node.outputs.thermal_dynamic_output.get_dict()[f'{cell_size}']
        leg_name_list.append(('Cell:' + cell_size.replace(' ', '_')))
        index = 0
        for i in line_dict.keys():
            line_for_plot[f"{cell_size.replace(' ', '_')}_{index}"] = line_dict[i]['specific_heat']
            line_name_list.append(f"{cell_size.replace(' ', '_')}_{index}")
            index = index + 1
    plot_line(
        node.inputs.temperatures.get_list(), line_for_plot, line_name_list, 'T', 'specific_heat',
        node.inputs.plot_dir.value, 'specific_heat_T', leg_name_list
    )


def plot_sus_phase_diagram(node: orm.Node):
    """Plot the magnetic susceptibility phase diagram"""
    y_label_list = []
    for i in node.inputs.external_fields.get_list():
        y_label_list.append(float(max(np.array(i.split()))))

    for _, cell_size in enumerate(node.inputs.cell_size):
        pd_dict = node.outputs.thermal_dynamic_output.get_dict()[f'{cell_size}']
        pd_for_plot = []
        for i in pd_dict.keys():
            pd_for_plot.append(pd_dict[i]['susceptibility'])
        plot_pd(
            node.inputs.plot_dir.value, pd_for_plot, node.inputs.temperatures.get_list(), y_label_list,
            ('susceptibility_T' + str(cell_size)), 'B', 'T'
        )


def plot_susceptibility_temp(node: orm.Node):
    """Plot the susceptibility as a function of temperature"""
    line_name_list = []
    leg_name_list = []
    line_for_plot = {}
    for _, cell_size in enumerate(node.inputs.cell_size):
        line_dict = node.outputs.thermal_dynamic_output.get_dict()[f'{cell_size}']
        leg_name_list.append(('Cell:' + cell_size.replace(' ', '_')))
        index = 0
        for i in line_dict.keys():
            line_for_plot[f"{cell_size.replace(' ', '_')}_{index}"] = line_dict[i]['susceptibility']
            line_name_list.append(f"{cell_size.replace(' ', '_')}_{index}")
            index = index + 1
    plot_line(
        node.inputs.temperatures.get_list(), line_for_plot, line_name_list, 'T', 'Susceptibility',
        node.inputs.plot_dir.value, 'Susceptibility_T', leg_name_list
    )


def plot_free_e_phase_diagram(node: orm.Node):
    """Plot the free energy phase diagram"""
    y_label_list = []
    for i in node.inputs.external_fields.get_list():
        y_label_list.append(float(max(np.array(i.split()))))

    for _, cell_size in enumerate(node.inputs.cell_size):
        pd_dict = node.outputs.thermal_dynamic_output.get_dict()[f'{cell_size}']
        pd_for_plot = []
        for i in pd_dict.keys():
            pd_for_plot.append(pd_dict[i]['free_e'])
        plot_pd(
            node.inputs.plot_dir.value,
            pd_for_plot,
            node.inputs.temperatures.get_list(),
            y_label_list,
            ('free_e_T' + str(cell_size)),
            'B',
            'T',
        )


def plot_free_e_temp(node: orm.Node):
    """Plot the free energy as a function of temperature"""
    line_name_list = []
    leg_name_list = []
    line_for_plot = {}
    for _, cell_size in enumerate(node.inputs.cell_size):
        line_dict = node.outputs.thermal_dynamic_output.get_dict()[f'{cell_size}']
        leg_name_list.append(('Cell:' + cell_size.replace(' ', '_')))
        index = 0
        for i in line_dict.keys():
            line_for_plot[f"{cell_size.replace(' ', '_')}_{index}"] = line_dict[i]['free_e']
            line_name_list.append(f"{cell_size.replace(' ', '_')}_{index}")
            index = index + 1
    plot_line(
        node.inputs.temperatures.get_list(), line_for_plot, line_name_list, 'T', 'free_e', node.inputs.plot_dir.value,
        'free_e_T', leg_name_list
    )


def plot_entropy_phase_diagram(node: orm.Node):
    """Plot the entropy phase diagram"""
    y_label_list = []
    for i in node.inputs.external_fields.get_list():
        y_label_list.append(float(max(np.array(i.split()))))

    for _, cell_size in enumerate(node.inputs.cell_size):
        pd_dict = node.outputs.thermal_dynamic_output.get_dict()[f'{cell_size}']
        pd_for_plot = []
        for i in pd_dict.keys():
            pd_for_plot.append(pd_dict[i]['entropy'])
        plot_pd(
            node.inputs.plot_dir.value, pd_for_plot, node.inputs.temperatures.get_list(), y_label_list,
            ('entropy_T' + str(cell_size)), 'B', 'T'
        )


def plot_entropy_temp(node: orm.Node):
    """Plot the entropy vs temperature"""
    line_name_list = []
    leg_name_list = []
    line_for_plot = {}
    for _, cell_size in enumerate(node.inputs.cell_size):
        line_dict = node.outputs.thermal_dynamic_output.get_dict()[f'{cell_size}']
        leg_name_list.append(('Cell:' + cell_size.replace(' ', '_')))
        index = 0
        for i in line_dict.keys():
            line_for_plot[f"{cell_size.replace(' ', '_')}_{index}"] = line_dict[i]['entropy']
            line_name_list.append(f"{cell_size.replace(' ', '_')}_{index}")
            index = index + 1
    plot_line(
        node.inputs.temperatures.get_list(), line_for_plot, line_name_list, 'T', 'Entropy', node.inputs.plot_dir.value,
        'Entropy_T', leg_name_list
    )


def plot_energy_phase_diagram(node: orm.Node):
    """Plot the energy phase diagram"""
    y_label_list = []
    for i in node.inputs.external_fields.get_list():
        y_label_list.append(float(max(np.array(i.split()))))

    for _, cell_size in enumerate(node.inputs.cell_size):
        pd_dict = node.outputs.thermal_dynamic_output.get_dict()[f'{cell_size}']
        pd_for_plot = []
        for i in pd_dict.keys():
            pd_for_plot.append(pd_dict[i]['energy'])
        plot_pd(
            node.inputs.plot_dir.value, pd_for_plot, node.inputs.temperatures.get_list(), y_label_list,
            ('energy_T' + str(cell_size)), 'B', 'T'
        )


def plot_energy_temp(node: orm.Node):
    """Plot the energy vs temperature"""
    line_name_list = []
    leg_name_list = []
    line_for_plot = {}
    for _, cell_size in enumerate(node.inputs.cell_size):
        line_dict = node.outputs.thermal_dynamic_output.get_dict()[f'{cell_size}']
        leg_name_list.append(('Cell:' + cell_size.replace(' ', '_')))
        index = 0
        for i in line_dict.keys():
            line_for_plot[f"{cell_size.replace(' ', '_')}_{index}"] = line_dict[i]['energy']
            line_name_list.append(f"{cell_size.replace(' ', '_')}_{index}")
            index = index + 1
    plot_line(
        node.inputs.temperatures.get_list(), line_for_plot, line_name_list, 'T', 'energy', node.inputs.plot_dir.value,
        'energy_T', leg_name_list
    )


def plot_dudt_phase_diagram(node: orm.Node):
    """Plot the variation of the energy phase diagram"""
    y_label_list = []
    for i in node.inputs.external_fields.get_list():
        y_label_list.append(float(max(np.array(i.split()))))

    for _, cell_size in enumerate(node.inputs.cell_size):
        pd_dict = node.outputs.thermal_dynamic_output.get_dict()[f'{cell_size}']
        pd_for_plot = []
        for i in pd_dict.keys():
            pd_for_plot.append(pd_dict[i]['dudt'])
        plot_pd(
            node.inputs.plot_dir.value, pd_for_plot, node.inputs.temperatures.get_list(), y_label_list,
            ('dudt_T' + str(cell_size)), 'B', 'T'
        )


def plot_dudt_temp(node: orm.Node):
    """Plot the variation of the energy vs temperature"""
    line_name_list = []
    leg_name_list = []
    line_for_plot = {}
    for _, cell_size in enumerate(node.inputs.cell_size):
        line_dict = node.outputs.thermal_dynamic_output.get_dict()[f'{cell_size}']
        leg_name_list.append(('Cell:' + cell_size.replace(' ', '_')))
        index = 0
        for i in line_dict.keys():
            line_for_plot[f"{cell_size.replace(' ', '_')}_{index}"] = line_dict[i]['dudt']
            line_name_list.append(f"{cell_size.replace(' ', '_')}_{index}")
            index = index + 1
    plot_line(
        node.inputs.temperatures.get_list(),
        line_for_plot,
        line_name_list,
        'T',
        'dudt',
        node.inputs.plot_dir.value,
        'dudt_T',
        leg_name_list,
    )


def plot_binder_cumu_phase_diagram(node: orm.Node):
    """Plot the binder cumulant phase diagram"""
    y_label_list = []
    for i in node.inputs.external_fields.get_list():
        y_label_list.append(float(max(np.array(i.split()))))

    for _, cell_size in enumerate(node.inputs.cell_size):
        pd_dict = node.outputs.thermal_dynamic_output.get_dict()[f'{cell_size}']
        pd_for_plot = []
        for i in pd_dict.keys():
            pd_for_plot.append(pd_dict[i]['binder_cumulant'])
        plot_pd(
            node.inputs.plot_dir.value, pd_for_plot, node.inputs.temperatures.get_list(), y_label_list,
            ('binder_cumulant_T' + str(cell_size)), 'B', 'T'
        )


def plot_binder_cumulant_temp(node: orm.Node):
    """Plot the binder cumulant as as function of temperature"""
    line_name_list = []
    leg_name_list = []
    line_for_plot = {}
    for _, cell_size in enumerate(node.inputs.cell_size):
        line_dict = node.outputs.thermal_dynamic_output.get_dict()[f'{cell_size}']
        leg_name_list.append(('Cell:' + cell_size.replace(' ', '_')))
        index = 0
        for i in line_dict.keys():
            line_for_plot[f"{cell_size.replace(' ', '_')}_{index}"] = line_dict[i]['binder_cumulant']
            line_name_list.append(f"{cell_size.replace(' ', '_')}_{index}")
            index = index + 1
    plot_line(
        node.inputs.temperatures.get_list(),
        line_for_plot,
        line_name_list,
        'T',
        'binder_cumulant',
        node.inputs.plot_dir.value,
        'binder_cumulant_T',
        leg_name_list,
    )
