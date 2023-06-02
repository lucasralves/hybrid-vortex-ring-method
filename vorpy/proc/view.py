import typing as tp
import vtkmodules.all as vtk

from vorpy.model.mesh import Mesh


def view(meshes: tp.List[Mesh], save_to: str):

    # Create pvd file
    time = [meshes[0][i]['time'] for i in range(len(meshes[0]))]
    instants = [i + 1 for i in range(len(meshes[0]))]

    file = open(save_to + 'main.pvd', 'w')
    file.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian" header_type="UInt64">\n')
    file.write('  <Collection>\n')
    for i in range(len(meshes[0])):
        file.write('    <DataSet timestep="{:.2f}" part="0" file="{}.vtp"/>\n'.format(time[i], instants[i]))
    file.write('  </Collection>\n')
    file.write(' </VTKFile>')
    file.close()

    # Create vtp files
    for i in range(len(meshes[0])):

        pd = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        lines = vtk.vtkCellArray()

        initial_index = 0

        for j in range(len(meshes)):

            mesh: Mesh = meshes[j][i]['mesh']

            for vt in mesh.vt:
                points.InsertNextPoint(vt[0], vt[1], vt[2])
            
            for fc in mesh.fc[:mesh.n_span, :]:
                cell = vtk.vtkQuad()
                cell.GetPointIds().SetId(0, initial_index + fc[0])
                cell.GetPointIds().SetId(1, initial_index + fc[1])
                cell.GetPointIds().SetId(2, initial_index + fc[2])
                cell.GetPointIds().SetId(3, initial_index + fc[3])
                cells.InsertNextCell(cell)
            
            if mesh.fc.shape[0] > mesh.n_span:
                for fc in mesh.fc[mesh.n_span:, :]:
                    polyLine = vtk.vtkPolyLine()
                    polyLine.GetPointIds().SetNumberOfIds(5)
                    polyLine.GetPointIds().SetId(0, initial_index + fc[0])
                    polyLine.GetPointIds().SetId(1, initial_index + fc[1])
                    polyLine.GetPointIds().SetId(2, initial_index + fc[2])
                    polyLine.GetPointIds().SetId(3, initial_index + fc[3])
                    polyLine.GetPointIds().SetId(4, initial_index + fc[0])
                    lines.InsertNextCell(polyLine)

            initial_index += mesh.vt.shape[0]
            
        pd.SetPoints(points)
        pd.SetPolys(cells)
        pd.SetLines(lines)
    
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName('{}{}.vtp'.format(save_to, instants[i]))
        writer.SetInputData(pd)
        writer.Write()

    return