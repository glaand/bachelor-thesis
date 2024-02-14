import pymesh

mesh = pymesh.load_mesh("mercedes_benz.obj")

grid = pymesh.VoxelGrid(0.5, mesh.dim);
grid.insert_mesh(mesh);
grid.create_grid();
grid.dilate(0);
grid.erode(0);
out_mesh = grid.mesh;
pymesh.save_mesh("voxel.obj", out_mesh);