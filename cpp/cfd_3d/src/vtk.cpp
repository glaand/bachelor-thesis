#include "cfd.h"
#include <vtkSmartPointer.h>
#include <vtkStructuredGrid.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPoints.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkXMLStructuredGridWriter.h>

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
    // @todo
}
