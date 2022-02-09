# Python script to visualize node sets and vectors from the frosch framework

from enum import Enum
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Button
from numpy.core.numeric import zeros_like
from numpy.lib.function_base import meshgrid
import meshio

# Read files
def read_meta():
    meta = np.loadtxt("meta.txt", dtype="int")
    nx, ny, processes = meta
    return nx, ny, processes

def get_appendix(process, name, iteration):
    name = "_"+name if name!="" else ""
    part = "_p" + str(process)+"_it"+"{:04d}".format(iteration)
    return name+part+".txt"

def nodes_from_txt(process, name="", iteration=0):
    appendix = get_appendix(process, name, iteration)
    nodes =  np.loadtxt("nodes"+appendix, dtype='int')

def values_from_txt(process, name="", iteration=0):
    appendix = get_appendix(process, name, iteration)
    values = np.loadtxt("values"+appendix, dtype="double")

def nodes_and_values_from_txt(process, name="", iteration=0):
    try:
        appendix = get_appendix(process, name, iteration)
        nodes =  np.loadtxt("nodes"+appendix, dtype='int')
        values = np.loadtxt("values"+appendix, dtype="double")
        return nodes, values
    except Exception as e:
        print(e)
        return None, None


# Grid functions
def generate_grid(nx,ny):
    x = np.linspace(0,1,nx)
    y = np.linspace(0,1,ny) 
    xx, yy = np.meshgrid(x,y)
    xx = xx.flatten()
    yy = yy.flatten()
    return x, y, xx, yy

def quad_connectivity(i, j, nx, ny):
        return [
            i + j * (nx + 1),
            i + 1 + j * (nx + 1),
            i + 1 + (j + 1) * (nx + 1),
            i + (j + 1) * (nx + 1),
        ]
# prolongation and boundary
def owned_nodes_mask(owned_nodes, size_full):
    mask = np.ones(size_full, dtype=bool)
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
def append_to_data_set(data_set, processes, name, iteration, preprocessor=lambda nodes, values: values):

    values_extended_list = []
    for process in range(0, processes):
        nodes, values = nodes_and_values_from_txt(process,name, iteration)
        values_extended = preprocessor(nodes, values)
        values_extended_list.append(values_extended)
        data_set[name+str(process)]= values_extended
    data_set[name]=np.sum(values_extended_list,axis=0)

def get_preprocessor(nx, ny, add_boundary, custom_func=None):
    if add_boundary:
            pre = lambda nodes, values : add_boundar_to_extended(nodes, prolongate(nodes, values, nx*ny), nx, ny)
    else:
        pre = lambda nodes, values : prolongate(nodes, values, nx*ny)
    
    if custom_func is None:
        preprocessor = pre
    else:
        preprocessor = lambda nodes, values: custom_func(pre(nodes, values))
    return preprocessor

def export_vtk(field_names, add_boundary=True, iterations=1):
    nx,ny,processes = read_meta()
    nb = 2 if add_boundary else 0
    mx=nx+nb
    my=ny+nb
    x,y,xx,yy = generate_grid(mx,my)
    
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
    for it in range(0,iterations):
        point_data = {}
        
        for field_name in field_names:
            preprocessor = get_preprocessor(nx, ny, add_boundary, np.abs if field_name in ["res", "global"] else None)
            try:
                append_to_data_set(point_data, processes, field_name, it, preprocessor)
            except Exception as e:
                print(e)
        
        mesh = meshio.Mesh(
            points,
            cells,
            point_data=point_data,
        )
        field_names_str = re.sub('[^A-Za-z0-9]+', '', str(field_names))
        filename_out = field_names_str+"_"+"{:04d}".format(it)+".vtk"
        mesh.write(filename_out)

def export_xdmf(field_names, add_boundary=True, iterations=1):
    nx,ny,processes = read_meta()
    nb = 2 if add_boundary else 0
    mx=nx+nb
    my=ny+nb
    x,y,xx,yy = generate_grid(mx,my)
    
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
    field_names_str = re.sub('[^A-Za-z0-9]+', '', str(field_names))
    filename_out = field_names_str+".xdmf"
    with meshio.xdmf.TimeSeriesWriter(filename_out) as writer:
        writer.write_points_cells(points, cells)
        for it in range(0,iterations):
            point_data = {}
            for field_name in field_names:
                preprocessor = get_preprocessor(nx, ny, True, np.abs if field_name in ["res", "global"] else None)
                try:
                    append_to_data_set(point_data, processes, field_name, it, preprocessor)
                except Exception as e:
                    print(e)
        
            writer.write_data(it, point_data=point_data)

# visualize
def visualizeNodes(grid, GOs: np.array, offset=np.array([0,0]), s=None, c=None, marker=None,label=None, alpha=0.3):
    xx, yy = grid
    plt.scatter(xx[GOs]+offset[0],yy[GOs]+offset[1], s,c,marker, alpha=alpha, label=label)

def visualizeValues(grid,GOs: np.array, values: np.array):
    xx, yy = grid
    i=0
    for x, y in zip(xx[GOs], yy[GOs]): 
        plt.text(x, y, f'{values[i]:.2}', color="red", fontsize=12)
        i=i+1

def visualizeVector(name:str, size=None, color=None, marker:str=None, offset=None, process_list=None, alpha=None):
    nx, ny, processes = read_meta()
    x,y,xx,yy = generate_grid(nx,ny)
    grid = (xx,yy)
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
            visualizeNodes(grid, nodes, offsets[process], s=sizes[process], c=colors[process], marker=marker, alpha=alphas[process], label=name+f"-p{process}")
            #visualizeValues(nodes, values)
            #plt.title("process"+str(process))
            #plt.title(name)
            #plt.show(block=False)
    legend = plt.legend(loc='upper right', fancybox=True, shadow=True)

def get_max_iterations():
    files = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith(".txt")]
    max = 0
    for f in files:
        try:
            it = int(str(f[-8:-4]))
            if it>max:
                max = it
        except:
            pass
    return max

if __name__=="__main__":
    max_iterations = get_max_iterations()
    print(max_iterations)
    vecs = ["w","cut"]
    markers=[None, "<"]
    field_names = ["global","res"]#"res", "sol",
    export_vtk(field_names, add_boundary=True, iterations=9)
    export_xdmf(field_names, add_boundary=True, iterations=9)
    plt.figure()
    for i,name in enumerate(vecs):
        visualizeVector(name, marker=markers[i],process_list=[0,1,2])
        plt.show(block=False)
    visualizeVector("unique", offset=[0,0], marker="x", size=5**2,process_list=[0,1,2], alpha=1.0)
    plt.title("nodes")
    #plt.show()
    
    