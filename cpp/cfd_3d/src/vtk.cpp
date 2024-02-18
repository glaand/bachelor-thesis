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
    vtk_grid->SetDimensions(sim->imax, sim->jmax, sim->kmax);

    // Create vtk points
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    for (int k = 0; k < sim->grid.kmax; ++k) {
        for (int j = 0; j < sim->jmax; ++j) {
            for (int i = 0; i < sim->imax; ++i) {
                points->InsertNextPoint(i * sim->grid.dx, j * sim->grid.dy, k * sim->grid.dz);
            }
        }
    }

    vtk_grid->SetPoints(points);

    // Add pressure data to the grid
    vtkSmartPointer<vtkDoubleArray> pressure_array = vtkSmartPointer<vtkDoubleArray>::New();
    pressure_array->SetNumberOfComponents(1);
    pressure_array->SetName("Pressure");

    // Add velocity data to the grid as a vector field
    vtkSmartPointer<vtkDoubleArray> velocity_array = vtkSmartPointer<vtkDoubleArray>::New();
    velocity_array->SetNumberOfComponents(3);  // 3 components for the vector (VelocityX, VelocityY, VelocityZ)
    velocity_array->SetComponentName(0, "VelocityX");
    velocity_array->SetComponentName(1, "VelocityY");
    velocity_array->SetComponentName(2, "VelocityZ");
    velocity_array->SetName("Velocity");

    for (int k = 1; k < sim->grid.kmax + 1; ++k) {
        for (int j = 1; j < sim->jmax + 1; ++j) {
            for (int i = 1; i < sim->imax + 1; ++i) {
                pressure_array->InsertNextValue(sim->grid.p.coeffRef(i, j, k));
                velocity_array->InsertNextTuple3(sim->grid.u_interpolated.coeffRef(i, j, k),
                                                 sim->grid.v_interpolated.coeffRef(i, j, k),
                                                 sim->grid.w_interpolated.coeffRef(i, j, k));
            }
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
    int startK = 0x7FFFFFFF;
    int endI = 0;
    int endJ = 0;
    int endK = 0;

    // Create vtk points
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    for (int k = 0; k < sim->grid.kmax; ++k) {
        for (int j = 0; j < sim->jmax; ++j) {
            for (int i = 0; i < sim->imax; ++i) {
                points->InsertNextPoint(i * sim->grid.dx, j * sim->grid.dy, k * sim->grid.dz);
            }
        }
    }

    for (int k = 1; k < sim->grid.kmax; ++k) {
        for (int j = 1; j < sim->jmax; ++j) {
            for (int i = 1; i < sim->imax; ++i) {
                if (sim->isObstacleCell(i, j, k)) {
                    startI = std::min(startI, i);
                    startJ = std::min(startJ, j);
                    startK = std::min(startK, k);
                    endI = std::max(endI, i);
                    endJ = std::max(endJ, j);
                    endK = std::max(endK, k);
                }
            }
        }
    }

    std::cout << "startI: " << startI << " startJ: " << startJ << " startK: " << startK << std::endl;

    // Create vtk cells for the geometry (e.g., a cube)
    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
    vtkIdType vertexIds[8];
    
    for (int k = startK; k < endK; ++k) {
        for (int j = startJ; j < endJ; ++j) {
            for (int i = startI; i < endI; ++i) {
                vertexIds[0] = i + sim->imax * j + sim->imax * sim->jmax * k;
                vertexIds[1] = (i + 1) + sim->imax * j + sim->imax * sim->jmax * k;
                vertexIds[2] = (i + 1) + sim->imax * (j + 1) + sim->imax * sim->jmax * k;
                vertexIds[3] = i + sim->imax * (j + 1) + sim->imax * sim->jmax * k;
                vertexIds[4] = i + sim->imax * j + sim->imax * sim->jmax * (k + 1);
                vertexIds[5] = (i + 1) + sim->imax * j + sim->imax * sim->jmax * (k + 1);
                vertexIds[6] = (i + 1) + sim->imax * (j + 1) + sim->imax * sim->jmax * (k + 1);
                vertexIds[7] = i + sim->imax * (j + 1) + sim->imax * sim->jmax * (k + 1);
                
                cells->InsertNextCell(8, vertexIds);
            }
        }
    }

    vtk_grid->SetPoints(points);
    vtk_grid->SetCells(VTK_HEXAHEDRON, cells);

    vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
    char filename[100];
    sprintf(filename, "geometry.vtu", sim->it);
    writer->SetFileName(filename);
    writer->SetInputData(vtk_grid);
    writer->Write();
}