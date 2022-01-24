# Python script to visualize node sets and vectors from the frosch framework

from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Button
from numpy.core.numeric import zeros_like
from numpy.lib.function_base import meshgrid
import meshio
from py import process

# Read files
def read_meta():
    meta = np.loadtxt("meta.txt", dtype="int")
    nx, ny, processes = meta
    return nx, ny, processes

def nodes_and_values_from_txt(process, name=""):
    try:
        nodes =  np.loadtxt("nodes_"+name+str(process)+".txt", dtype='int')
        values = np.loadtxt("values_"+name+str(process)+".txt", dtype="double")
        return nodes, values
    except Exception as e:
        print(e)
        return None, None

# Grid functions
def generate_grid(nx,ny):
    global x,y,xx,yy
    x = np.linspace(0,1,nx)
    y = np.linspace(0,1,ny) 
    xx, yy = np.meshgrid(x,y)
    xx = xx.flatten()
    yy = yy.flatten()

def quad_connectivity(i, j, nx, ny):
        return [
            i + j * (nx + 1),
            i + 1 + j * (nx + 1),
            i + 1 + (j + 1) * (nx + 1),
            i + (j + 1) * (nx + 1),
        ]
# prolongation and boundary
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

#export
def export_vtk(add_boundary=True,name=""):
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
        nodes, values = nodes_and_values_from_txt(process,name)
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

# visualize
def visualizeNodes(GOs: np.array, offset=np.array([0,0]), s=None, c=None, marker=None,label=None, alpha=0.3):
    plt.scatter(xx[GOs]+offset[0],yy[GOs]+offset[1], s,c,marker, alpha=alpha, label=label)

def visualizeValues(GOs: np.array, values: np.array):
    i=0
    for x, y in zip(xx[GOs], yy[GOs]): 
        plt.text(x, y, f'{values[i]:.2}', color="red", fontsize=12)
        i=i+1

def visualizeVector(name:str, size=None, color=None, marker:str=None, offset=None, process_list=None, alpha=None):
    nx, ny, processes = read_meta()
    generate_grid(nx,ny)
    plt.scatter(xx,yy, s=1, color="black")

    if process_list is None:
        process_list = range(0,processes)
    else:
        processes=len(process_list)

    if color is None:
        colors = ["r","g","b","y"]
    else:
        colors = [color]*processes
    if size is None:
        sizes = [15**2]*processes
    else:
        sizes = [size]*processes
    #offsets=np.array([[0.,1./ny], [0.,-1./ny],[1./ny,0.],[-1./ny,0.]])
    if offset is None:
        offsets=np.array([[0.1/nx,0.1/ny], [-.1/nx,-.1/ny],[-.1/nx,.1/ny],[.1/nx,-.1/ny]])#TODO: for 9 processes?
    else:
        offsets=np.array([offset]*processes)
    if alpha is None:
        alphas = [0.3]*processes
    else:
        alphas = [alpha]*processes

    for process in process_list:
        nodes, values = nodes_and_values_from_txt(process,name)
        if nodes is not None:
            visualizeNodes(nodes, offsets[process], s=sizes[process], c=colors[process], marker=marker, alpha=alphas[process], label=name+f"-p{process}")
            #visualizeValues(nodes, values)
            #plt.title("process"+str(process))
            plt.title(name)
            #plt.show(block=False)
    legend = plt.legend(loc='upper right', fancybox=True, shadow=True)

if __name__=="__main__": 
    vecs = ["w","cut"]
    markers=[None, "<"]
    export_vtk(add_boundary=True, name=vecs[0])
    plt.figure()
    for i,name in enumerate(vecs):
        visualizeVector(name, marker=markers[i],process_list=[0,1,2])
        plt.show(block=False)
    visualizeVector("unique", offset=[0,0], marker="x", size=5**2,process_list=[0,1,2], alpha=1.0)
    plt.show()
    
    