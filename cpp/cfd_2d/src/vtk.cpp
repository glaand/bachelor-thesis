#include "cfd.h"
#include <vtkSmartPointer.h>
#include <vtkStructuredGrid.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPoints.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkXMLStructuredGridWriter.h>
#include <vtkXMLUnstructuredGridWriter.h>

void CFD::saveVTK(FluidSimulation* sim) {
    vtkSmartPointer<vtkStructuredGrid> vtk_grid = vtkSmartPointer<vtkStructuredGrid>::New();
    vtk_grid->SetDimensions(sim->imax, sim->jmax, 1);

    // Create vtk points
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    for (int j = 0; j < sim->jmax; ++j) {
        for (int i = 0; i < sim->imax; ++i) {
            points->InsertNextPoint(i * sim->grid.dx, j * sim->grid.dy, 0);
        }
    }

    vtk_grid->SetPoints(points);

    // Add pressure data to the grid
    vtkSmartPointer<vtkDoubleArray> pressure_array = vtkSmartPointer<vtkDoubleArray>::New();
    pressure_array->SetNumberOfComponents(1);
    pressure_array->SetName("Pressure");

    // Add velocity data to the grid as a vector field
    vtkSmartPointer<vtkDoubleArray> velocity_array = vtkSmartPointer<vtkDoubleArray>::New();
    velocity_array->SetNumberOfComponents(3);  // 3 components for the vector (VelocityX, VelocityY, 0)
    velocity_array->SetComponentName(0, "VelocityX");
    velocity_array->SetComponentName(1, "VelocityY");
    velocity_array->SetComponentName(2, "VelocityZ");  // Z component is set to 0
    velocity_array->SetName("Velocity");

    for (int j = 0; j < sim->jmax; ++j) {
        for (int i = 0; i < sim->imax; ++i) {
            pressure_array->InsertNextValue(sim->grid.p.coeffRef(i, j));
            velocity_array->InsertNextTuple3(sim->grid.u_interpolated.coeffRef(i, j),
                                             sim->grid.v_interpolated.coeffRef(i, j),
                                             0.0);  // Z component is set to 0
        }
    }

    vtk_grid->GetPointData()->AddArray(pressure_array);
    vtk_grid->GetPointData()->AddArray(velocity_array);

    vtkSmartPointer<vtkXMLStructuredGridWriter> writer = vtkSmartPointer<vtkXMLStructuredGridWriter>::New();
    char filename[100];
    sprintf(filename, "output_%d.vts", sim->it);
    writer->SetFileName(filename);
    writer->SetInputData(vtk_grid);
    writer->Write();
}

void CFD::saveVTKGeometry(FluidSimulation* sim) {
    vtkSmartPointer<vtkUnstructuredGrid> vtk_grid = vtkSmartPointer<vtkUnstructuredGrid>::New();

    int startI = 0x7FFFFFFF;
    int startJ = 0x7FFFFFFF;
    int endI = 0;
    int endJ = 0;

    // Create vtk points
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    for (int j = 0; j < sim->jmax; ++j) {
        for (int i = 0; i < sim->imax; ++i) {
            points->InsertNextPoint(i * sim->grid.dx, j * sim->grid.dy, 0.0);  // Assuming 2D, setting z-coordinate to 0.0
        }
    }

    for (int j = 1; j < sim->jmax; ++j) {
        for (int i = 1; i < sim->imax; ++i) {
            if (sim->isObstacleCell(i, j)) {
                startI = std::min(startI, i);
                startJ = std::min(startJ, j);
                endI = std::max(endI, i);
                endJ = std::max(endJ, j);
            }
        }
    }

    // Create vtk cells for the geometry
    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
    vtkIdType vertexIds[4];  // Assuming 2D, use 4 vertices for each cell
    
    for (int j = startJ; j < endJ; ++j) {
        for (int i = startI; i < endI; ++i) {
            vertexIds[0] = i + sim->imax * j;
            vertexIds[1] = (i + 1) + sim->imax * j;
            vertexIds[2] = (i + 1) + sim->imax * (j + 1);
            vertexIds[3] = i + sim->imax * (j + 1);
            
            cells->InsertNextCell(4, vertexIds);  // Use 4 for quadrilateral cells in 2D
        }
    }

    vtk_grid->SetPoints(points);
    vtk_grid->SetCells(VTK_QUAD, cells);  // Use VTK_QUAD for 2D quadrilateral cells

    vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
    char filename[100];
    sprintf(filename, "geometry.vtu", sim->it);
    writer->SetFileName(filename);
    writer->SetInputData(vtk_grid);
    writer->Write();
}
