# Python script to visualize node sets and vectors from the frosch framework

from enum import Enum
from os import truncate
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Button
from numpy.core.numeric import zeros_like
from numpy.lib.function_base import meshgrid
import meshio

def read_meta():
    meta = np.loadtxt("meta.txt", dtype="int")
    nx, ny, processes = meta
    return nx, ny, processes

def generate_grid(nx,ny):
    global x,y,xx,yy
    x = np.linspace(0,1,nx)
    y = np.linspace(0,1,ny) 
    xx, yy = np.meshgrid(x,y)
    xx = xx.flatten()
    yy = yy.flatten()

def nodes_and_values_from_txt(process):
    try:
        nodes =  np.loadtxt("nodes"+str(process)+".txt", dtype='int')
        values = np.loadtxt("values"+str(process)+".txt", dtype="double")
        return nodes, values
    except Exception as e:
        print(e)
        return None, None

def quad_connectivity(i, j, nx, ny):
        return [
            i + j * (nx + 1),
            i + 1 + j * (nx + 1),
            i + 1 + (j + 1) * (nx + 1),
            i + (j + 1) * (nx + 1),
        ]

def export_vtk(add_boundary=True):
    nx,ny,processes = read_meta()
    if add_boundary:
        mx=nx+2
        my=ny+2
    else:
        mx=nx
        my=ny
    generate_grid(mx,my)
    points = np.vstack([np.ravel(xx), np.ravel(yy)]).T
    cells = [
        (
            "quad",
            [
                quad_connectivity(i, j, mx-1, my-1)
                for i in range(0, mx-1)
                for j in range(0, my-1)
            ],
        )
    ]
    point_data={}
    for process in range(0, processes):
        nodes, values = nodes_and_values_from_txt(process)
        values_extended = prolongate(nodes, values, nx*ny)
        if add_boundary:
            values_extended = add_boundar_to_extended(nodes, values_extended, nx, ny)
        point_data["u"+str(process)]= values_extended

    mesh = meshio.Mesh(
        points,
        cells,
        point_data=point_data,
    )
    mesh.write("out.vtk", file_format="vtk", binary=False)

def owned_nodes_mask(owned_nodes):
    mask = np.ones(x.size, dtype=bool)
    mask[owned_nodes]=False

def prolongate(GOs, values,ntot):
    values_full = np.zeros(ntot)
    values_full[GOs]=values
    return values_full

def add_boundar_to_extended(GOs, values, nx, ny):
    values_with_boundary = np.zeros((nx+2)*(ny+2))
    for go in GOs:
        i = go%nx
        j = int(go/nx)
        values_with_boundary[go+nx+2+2*j+1]=values[go]
    return values_with_boundary

def visualizeNodes(GOs: np.array):
    # basic grid
    plt.figure()
    plt.scatter(xx,yy,color="black")
    plt.scatter(xx[GOs],yy[GOs], color="red")

def visualizeValues(GOs: np.array, values: np.array):
    i=0
    for x, y in zip(xx[GOs], yy[GOs]): 
        plt.text(x, y, f'{values[i]:.2}', color="red", fontsize=12)
        i=i+1

def _2dview(event, ax):
    print("hey")
    ax.view_init(azim=0, elev=90)

def visualize3d(GOs: np.array, values:np.array):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(xx,yy,np.zeros_like(xx), color="black")
    ax.scatter3D(xx[GOs],yy[GOs], values, color="red")
    print("Hey")
    # # Allow 2d view
    # axcut = plt.axes([0.9, 0.0, 0.1, 0.075])
    # bcut = Button(axcut, 'YES', color='red', hovercolor='green')
    # bcut.on_clicked(_2dview)#lambda event: _2dview(event, ax,GOs, values))

def visualizeArray(GOs: np.array, values: np.array):
    mask = owned_nodes_mask(GOs)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(xx[GOs],yy[GOs],values, cmap=cm.jet, linewidth=0.2)
    surf = ax.plot_trisurf(xx[mask],yy[mask], values_full[mask], cmap=cm.jet, linewidth=0.2)
    fig.colorbar(surf)

def visualize(coords, map, values_vector=None):
    #TODO
    return


if __name__=="__main__":
    export_vtk()
    twoD = True
    nx, ny, processes = read_meta()
    print(nx,ny,processes)
    generate_grid(nx,ny)
    for process in range(0,processes):
        nodes, values = nodes_and_values_from_txt(process)
        print(nodes, values)
        if nodes is not None:
            if twoD:
                visualizeNodes(nodes)
                visualizeValues(nodes, values)
            else:
                visualize3d(nodes, values)
            plt.title("process"+str(process))
            plt.show(block=False)
    plt.show()


#axcut = plt.axes([0.9, 0.0, 0.1, 0.075])
#bcut = Button(axcut, 'hey', color='red', hovercolor='green')
#bcut.on_clicked(lambda event: _2dview(event, ax))   