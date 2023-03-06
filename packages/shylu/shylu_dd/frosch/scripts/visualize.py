# Python script to visualize xpetra maps and vectors, specialized to domain decomposition from the frosch framework

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

from tikzplotlib import save as tikz_save

########################################################################### Read files written out by the c++ code
def read_meta():
    """Load the meta data (nx, ny, processes) from the meta.txt file."""
    meta = np.loadtxt("meta.txt", dtype="int")
    nx, ny, processes = meta
    return nx, ny, processes

def get_appendix(process, name, iteration=None):
    """Create the filename appendix for the given parameters.

    Args:
        process (int): The number of the output process.
        name (str): The name of the map/vector.
        iteration (int, optional): The iteration count. Defaults to None.

    Returns:
        _type_: _description_
    """
    name = "_"+name if name!="" else ""
    part = "_p" + str(process)
    if iteration is not None:
        part = part + "_it"+"{:04d}".format(iteration)
    return name+part+".txt"

def nodes_from_txt(process, name, iteration=None):
    """Read a set of nodes from a file, which contains the xpetra map on one process."""
    appendix = get_appendix(process, name, iteration)
    fname = "nodes"+appendix
    nodes =  np.loadtxt(fname, dtype='int')
    return nodes

def values_from_txt(process, name, iteration=None):
    """Read a set of values from a file, which contains the xpetra vector on one process."""
    appendix = get_appendix(process, name, iteration)
    fname = "values"+appendix
    values = np.loadtxt(fname, dtype="double")
    return values

def nodes_and_values_from_txt(process, name="", iteration=None):
    """Read a set of nodes and values from a file, which contains the xpetra vector on one process."""
    try:
        appendix = get_appendix(process, name, iteration)
        nodes =  np.loadtxt("nodes"+appendix, dtype='int')
        values = np.loadtxt("values"+appendix, dtype="double")
        return nodes, values
    except Exception as e:
        print(e)
        return None, None


########################################################################### Grid functions
def generate_grid(nx,ny):
    """Generate a uniform grid of size nx*ny."""
    x = np.linspace(0,1,nx)
    y = np.linspace(0,1,ny) 
    xx, yy = np.meshgrid(x,y)
    xx = xx.flatten()
    yy = yy.flatten()
    return x, y, xx, yy

def quad_connectivity(i, j, nx, ny):
    """Generate the connectivity for a quad element with index (i, j) in a uniform grid of size nx*ny."""
    return [
        i + j * (nx + 1),
        i + 1 + j * (nx + 1),
        i + 1 + (j + 1) * (nx + 1),
        i + (j + 1) * (nx + 1),
    ]

def prolongate(GOs, values,ntot):
    """Prolongate a vector with the given global indices and values to a vector of size ntot."""
    values_full = np.zeros(ntot)
    values_full[GOs]=values
    return values_full

def add_boundar_to_extended(GOs, values, nx, ny):
    """Add a boundary to a vector with the given global indices and values. Only works for a uniform grid of size nx*ny."""
    values_with_boundary = np.zeros((nx+2)*(ny+2))
    for go in GOs:
        i = go%nx
        j = int(go/nx)
        values_with_boundary[go+nx+2+2*j+1]=values[go]
    return values_with_boundary

########################################################################### export
def append_to_data_set(data_set, processes, name, iteration, preprocessor=lambda nodes, values: values):
    """Reads the nodes and values for a field with the given name and the given iteration count, and saves them to the data_set.
    The data is preprocessed by the given preprocessor function, which is applied to the nodes and values of each process."""
    values_extended_list = []
    for process in range(0, processes):
        nodes, values = nodes_and_values_from_txt(process,name, iteration)
        values_extended = preprocessor(nodes, values)
        values_extended_list.append(values_extended)
        data_set[name+str(process)]= values_extended
    data_set[name]=np.sum(values_extended_list,axis=0)

def get_preprocessor(nx, ny, add_boundary, custom_func=None):
    """Returns a preprocessor function, which can be used to prolongate a vector to the full domain, eventually add a boundary and apply a custom function to the result."""
    if add_boundary:
            pre = lambda nodes, values : add_boundar_to_extended(nodes, prolongate(nodes, values, nx*ny), nx, ny)
    else:
        pre = lambda nodes, values : prolongate(nodes, values, nx*ny)
    
    if custom_func is None:
        preprocessor = pre
    else:
        preprocessor = lambda nodes, values: custom_func(pre(nodes, values))
    return preprocessor

def export_vtk(fname, field_names, add_boundary=True, iterations=1):
    """Reads the nodes and values for a list of field names, and up to the iteration count and saved them to a file named fname in vtk format."""
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
        mesh.write(fname+"_"+"{:04d}".format(it)+".vtk")

def export_xdmf(fname, field_names, add_boundary=True, iterations=1):
    """Reads the nodes and values for a list of field names, and up to the iteration count and saved them to a file named fname in xdmf format."""
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
    with meshio.xdmf.TimeSeriesWriter(fname+".xdmf") as writer:
        writer.write_points_cells(points, cells)
        for it in range(0,iterations+1):
            point_data = {}
            for field_name in field_names:
                preprocessor = get_preprocessor(nx, ny, True, np.abs if field_name in ["res", "global"] else None)
                try:
                    append_to_data_set(point_data, processes, field_name, it, preprocessor)
                except Exception as e:
                    print(e)        

            writer.write_data(it, point_data=point_data)

########################################################################### Visualization with matplotlib
def visualizeNodes(grid, GOs: np.array, offset=np.array([0,0]), s=None, c=None, marker=None,label=None, alpha=0.3):
    """Visualizes a xpetra map by showing a scatter plot of the nodes."""
    xx, yy = grid
    plt.scatter(xx[GOs]+offset[0],yy[GOs]+offset[1], s,c,marker, alpha=alpha, label=label)

def visualizeValues(grid,GOs: np.array, values: np.array, color="black"):
    """Visualizes the values of a xpetra vector by plotting the values next to the corresponding nodes of the map."""
    # sanitize
    eps = 1e-8
    values_zeroed = np.where(np.abs(values)<eps, 0, values)
    xx, yy = grid
    i=0
    for x, y in zip(xx[GOs], yy[GOs]): 
        plt.text(x, y, f'{values_zeroed[i]:.2}', color=color, fontsize=10)
        i=i+1

def visualizeMap(name:str, size=None, color=None, marker:str=None, offset=None, process_list=None, alpha=None, showValues=False, it=None):
    """Visualizes a xpetra map by showing a scatter plot of the nodes and the values of a xpetra vector next to the nodes."""
    nx, ny, processes = read_meta()
    if process_list is None:
        process_list = range(0,processes)
    processes_plot = len(process_list)

    # plot basic grid
    x,y,xx,yy = generate_grid(nx,ny)
    grid = (xx,yy)
    plt.scatter(xx,yy, s=1, color="black")

    # Set all visual parameters
    my_var_len = 4 if processes_plot<=4 else 9
    if color is None:
        if processes_plot<=4:
            colors = ["r","g","b","y"]
        else:
            colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown", "tab:pink","tab:gray","tab:olive"]
    else:
        colors = [color]*processes_plot
    if offset is None:
        if processes_plot<=4:
            offsets=np.array([[0.1/nx,0.1/ny], [-.1/nx,-.1/ny],[-.1/nx,.1/ny],[.1/nx,-.1/ny]])
        else:
            offx = 0.15/nx
            offy = 0.15/ny
            offsets=np.array([[-offx,-offy],[0.,-offy],[offx,-offy],[-offx,0.0],[0.,0.],[offx,0.0],[-offx,offy],[0.,offy],[offx,offy]])
    else:
        offsets=np.array([offset]*processes)
    
    if size is None:
        sizes = [15**2]*processes_plot
    else:
        sizes = [size]*processes_plot
    if alpha is None:
        alphas = [0.3]*processes_plot
    else:
        alphas = [alpha]*processes_plot
    
    # Read in and plot
    try:
        for process_lin in range(0,processes_plot):
            set_i = process_lin % my_var_len
            process = process_list[process_lin]
            if showValues:
                nodes, values = nodes_and_values_from_txt(process,name, iteration=it)
            else:
                nodes = nodes_from_txt(process,name, iteration=it)
            if nodes is not None:
                visualizeNodes(grid, nodes, offsets[set_i], s=sizes[set_i], c=colors[set_i], marker=marker, alpha=alphas[set_i], label=name+f"-p{process}")
                if showValues:
                    visualizeValues(grid, nodes, values, color=colors[set_i])
        legend = plt.legend(loc='upper right', fancybox=True, shadow=True)
    except Exception as e:
        print(e)
        print(f"Didn't find files for name {name} {process}")


################################################################################## Main part
def get_max_iterations():
    """Returns the maximum iteration number of the current run by looking at the files in the current directory."""
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

import sys

if __name__=="__main__":
    export = True
    plot = False
    tex = False
    nx, ny, processes = read_meta()

    appendix = ""
    if len(sys.argv)>1:
        appendix = "_"+sys.argv[1]
    max_iterations = get_max_iterations()
    print(max_iterations)
    
    if export:
        field_names = ["rhs", "rhsHarmonic", "w", "res","sol","unique", "XOverlap_", "XOverlap_New","XTmp_","LocalSol","rhsPreSolveTmp_", "multiplicityExtended", "solution_final", "xExact"]
        field_names_str = re.sub('[^A-Za-z0-9]+', '', str(field_names))
        fname = "out_"+appendix
        #export_vtk(fname, field_names, add_boundary=True, iterations=max_iterations)
        export_xdmf(fname, field_names, add_boundary=True, iterations=max_iterations)

    if plot:
        markers=["$r$","o","s","$o$"]
        vecs = ["ovlp","nonOvlp", "interface"]
        #vecs=["overlapping", "interface"]#,"cut"]
        #vecs=["repeated","overlapping", "interface","ovlp"]
        #vecs=["interface","cut"]
        #vecs=["overlapping", "ovlp", "interface", "repeated"]
        process_list=[4,5,6,7,8]
        #process_list=[0,1,2,3,4,5,6,34,35,28,29]
        #process_list=[i for i in range(0,processes)]
        plt.figure()
        for i,name in enumerate(vecs):
            visualizeMap(name, marker=markers[i],process_list=process_list)
            plt.show(block=False)
        visualizeMap("unique", offset=[0,0], marker="o", size=5**2,process_list=process_list, alpha=1.0)
        visualizeMap("unique", offset=[0,0], marker="x", size=5**2,process_list=process_list, alpha=1.0)
        #visualizeMap("res", offset=[0,0], color="black", marker="x", size=5**2,process_list=process_list, alpha=1.0, showValues=True, it=0)
        #visualizeMap("XTmp_", offset=[0,0], color="black", marker="x", size=5**2,process_list=process_list, alpha=1.0, showValues=True, it=0) 
        plt.title("nodes "+appendix)
        plt.legend(bbox_to_anchor=(1.06,1.))
        plt.show(block=False)

    if tex:
        import matplotlib.patches as patches
        #vecs_old = ["ovlpOld", "innerOld", "interfaceOld", "cutOld", "multipleOld", "cut"]
        #vecs=["repeated","overlapping", "interface","ovlp", "inner", "restrDomain"]+ vecs_old
        vecs = ["overlapping", "ovlp", "inner", "interface", "restrDomain"]
        ploty = int(np.sqrt(len(vecs)))
        plotx = int(np.ceil(len(vecs)/ploty))
        process=4
        it=None

        # plot basic grid
        nx, ny, processes = read_meta()
        # nx=20
        # ny=10
        x,y,xx,yy = generate_grid(nx,ny)
        grid = (xx,yy)

        plt.figure()
        for i,name in enumerate(vecs):
            ax = plt.subplot(plotx, ploty, i+1)
            plt.scatter(xx,yy, s=1, color="black")
            try:
                nodes = nodes_from_txt(process,name, iteration=it)
                if nodes is not None:
                    visualizeNodes(grid, nodes, offset=[0,0], s=4**2, c="b", marker="o", alpha=1., label=name)
                    if name=="multRow" or name=="multCol" or name=="multiplicityExtended":
                        values = values_from_txt(process, name, iteration=it)
                        visualizeValues(grid, nodes, values)
                nodes = nodes_from_txt(process,"unique", iteration=it)
                ll = (np.min(xx[nodes])-1./nx/3, np.min(yy[nodes])-1./ny/3)
                ur = (np.max(xx[nodes])+1./nx/3, np.max(yy[nodes])+1./ny/3)
                wh = np.array(ur)-np.array(ll)
                rect = patches.Rectangle(ll, wh[0], wh[1], linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.set_xlim(0.2,0.8)
                ax.set_ylim(0.2,0.8)
                
                # if nodes is not None:
                #     visualizeNodes(grid, nodes, offset=[0,0], s=3**2,c="r", marker="o", alpha=0.7, label="unique")
                legend = plt.legend(loc='upper right', fancybox=True, shadow=True)
            except Exception as e:
                print(e)
                print(f"Didn't find files for name {name} {process}")
        plt.suptitle("Nodesets "+ appendix)
        tikz_save('NodeSets.tex')
        plt.show()

plt.show()