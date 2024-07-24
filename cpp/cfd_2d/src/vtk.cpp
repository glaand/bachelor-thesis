#include "cfd.h"
#include <vtkSmartPointer.h>
#include <vtkStructuredGrid.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPoints.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkXMLStructuredGridWriter.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <mpi.h>

void CFD::saveVTK(FluidSimulation* sim) {
    // Local grid dimensions excluding ghost cells
    int local_imax = sim->imax;
    int local_jmax = sim->jmax;
    int local_size = local_imax * local_jmax;
    int local_size_velocity = (local_imax + 1) * local_jmax;

    // Allocate memory for local data using Eigen matrices
    Eigen::MatrixXf local_pressure(local_imax, local_jmax);
    Eigen::MatrixXf local_velocity_u(local_imax, local_jmax + 1);
    Eigen::MatrixXf local_velocity_v(local_imax + 1, local_jmax);

    // Fill local matrices with simulation data excluding ghost cells
    for (int j = 1; j <= local_jmax; ++j) {
        for (int i = 1; i <= local_imax; ++i) {
            local_pressure(i - 1, j - 1) = sim->grid.p.coeffRef(i, j);
        }
    }

    for (int j = 1; j <= local_jmax + 1; ++j) {
        for (int i = 1; i <= local_imax; ++i) {
            local_velocity_u(i - 1, j - 1) = sim->grid.u.coeffRef(i, j);
        }
    }

    for (int j = 1; j <= local_jmax; ++j) {
        for (int i = 1; i <= local_imax + 1; ++i) {
            local_velocity_v(i - 1, j - 1) = sim->grid.v.coeffRef(i, j);
        }
    }

    if (sim->world_rank == 0) {
        // Calculate the global grid dimensions excluding ghost cells
        int global_imax = local_imax * sim->proc_grid_x;
        int global_jmax = local_jmax * sim->proc_grid_y;

        Eigen::MatrixXf global_velocity_u(local_imax * sim->proc_grid_x, (local_jmax + 1)  * sim->proc_grid_y);
        Eigen::MatrixXf global_velocity_v((local_imax + 1) * sim->proc_grid_x, local_jmax * sim->proc_grid_y);
        Eigen::MatrixXf u_interpolated(global_imax + 2, global_jmax + 2);
        Eigen::MatrixXf v_interpolated(global_imax + 2, global_jmax + 2);

        // Construct the global VTK grid
        vtkSmartPointer<vtkStructuredGrid> vtk_grid = vtkSmartPointer<vtkStructuredGrid>::New();
        vtk_grid->SetDimensions(global_imax, global_jmax, 1);

        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
        for (int j = 0; j < global_jmax; ++j) {
            for (int i = 0; i < global_imax; ++i) {
                points->InsertNextPoint(i * sim->grid.dx, j * sim->grid.dy, 0.0);
            }
        }
        vtk_grid->SetPoints(points);

        vtkSmartPointer<vtkFloatArray> pressure_array = vtkSmartPointer<vtkFloatArray>::New();
        pressure_array->SetNumberOfComponents(1);
        pressure_array->SetName("Pressure");
        pressure_array->SetNumberOfTuples(global_imax * global_jmax);

        vtkSmartPointer<vtkFloatArray> velocity_array = vtkSmartPointer<vtkFloatArray>::New();
        velocity_array->SetNumberOfComponents(3);
        velocity_array->SetName("Velocity");
        velocity_array->SetNumberOfTuples(global_imax * global_jmax);

        // Receive data from other processes and fill the global matrix
        for (int p = 0; p < sim->world_size; ++p) {
            Eigen::MatrixXf recv_pressure(local_imax, local_jmax);
            Eigen::MatrixXf recv_velocity_u(local_imax, local_jmax + 1);
            Eigen::MatrixXf recv_velocity_v(local_imax + 1, local_jmax);

            if (p == sim->world_rank) {
                recv_pressure = local_pressure;
                recv_velocity_u = local_velocity_u;
                recv_velocity_v = local_velocity_v;
            } else {
                MPI_Recv(recv_pressure.data(), local_size, MPI_FLOAT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_velocity_u.data(), local_size_velocity, MPI_FLOAT, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(recv_velocity_v.data(), local_size_velocity, MPI_FLOAT, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            int grid_x = p % sim->proc_grid_x;
            int grid_y = p / sim->proc_grid_x;
            int offset_x = grid_x * local_imax;
            int offset_y = grid_y * local_jmax;

            for (int j = 0; j < local_jmax; ++j) {
                for (int i = 0; i < local_imax; ++i) {
                    int global_idx = (offset_y + j) * global_imax + offset_x + i;
                    pressure_array->SetValue(global_idx, recv_pressure(i, j));
                }
            }

            for (int j = 0; j < local_jmax + 1; ++j) {
                for (int i = 0; i < local_imax; ++i) {
                    global_velocity_u(offset_x + i, offset_y + j) = recv_velocity_u(i, j);
                }
            }

            for (int j = 0; j < local_jmax; ++j) {
                for (int i = 0; i < local_imax + 1; ++i) {  
                    global_velocity_v(offset_x + i, offset_y + j) = recv_velocity_v(i, j);
                }
            }
        }

        for (int j = 1; j < global_jmax; j++) { // Start from 1 to exclude boundary cells
            for (int i = 1; i < global_imax; i++) { // Start from 1 to exclude boundary cells
                // Interpolate u velocity component at half-integer grid points
                float u1 = global_velocity_u(i, j);
                float u2 = (j + 1 < global_jmax) ? global_velocity_u(i, j + 1) : global_velocity_u(i, j);
                u_interpolated(i, j) = (u1 + u2) / 2;

                // Interpolate v velocity component at half-integer grid points
                float v1 = global_velocity_v(i, j);
                float v2 = (i + 1 < global_imax) ? global_velocity_v(i + 1, j) : global_velocity_v(i, j);
                v_interpolated(i, j) = (v1 + v2) / 2;

                // Fine interpolation along x direction
                float u3 = (i + 1 < global_imax) ? global_velocity_u(i + 1, j) : global_velocity_u(i, j);
                float u4 = (i + 1 < global_imax && j + 1 < global_jmax) ? global_velocity_u(i + 1, j + 1) : global_velocity_u(i, j);
                double delta_x = 0.5 * (u1 + u2 - u3 - u4);
                u_interpolated(i + 1, j) = global_velocity_u(i, j + 1) + delta_x;

                // Fine interpolation along y direction
                float v3 = (j + 1 < global_jmax) ? global_velocity_v(i, j + 1) : global_velocity_v(i, j);
                float v4 = (i + 1 < global_imax && j + 1 < global_jmax) ? global_velocity_v(i + 1, j + 1) : global_velocity_v(i, j);
                double delta_y = 0.5 * (v1 + v2 - v3 - v4);
                v_interpolated(i, j + 1) = global_velocity_v(i + 1, j) + delta_y;

                // Add interpolated velocity components to the VTK grid
                int global_idx = j * global_imax + i;
                velocity_array->SetTuple3(global_idx, u_interpolated(i, j), v_interpolated(i, j), 0.0);
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
    } else {
        // Send local data to the root process
        MPI_Send(local_pressure.data(), local_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(local_velocity_u.data(), local_size_velocity, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(local_velocity_v.data(), local_size_velocity, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
    }
}

void CFD::saveVTKGeometry(FluidSimulation* sim) {
    vtkSmartPointer<vtkUnstructuredGrid> vtk_grid = vtkSmartPointer<vtkUnstructuredGrid>::New();

    int startI = 0x7FFFFFFF;
    int startJ = 0x7FFFFFFF;
    int endI = 0;
    int endJ = 0;

    // Create vtk points
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    for (int j = 1; j < sim->jmax + 1; ++j) {
        for (int i = 1; i < sim->imax + 1; ++i) {
            points->InsertNextPoint(i * sim->grid.dx, j * sim->grid.dy, 0.0);  // Assuming 2D, setting z-coordinate to 0.0
        }
    }

    for (int j = 1; j < sim->jmax + 1; ++j) {
        for (int i = 1; i < sim->imax + 1; ++i) {
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