#!/usr/bin/env python
import os
import glob
import shlex
import shutil
import sys
import numpy as np
import subprocess

executable_path = "/faststorage/ianscluster08/kaiserls/Trilinos/build/packages/shylu/shylu_dd/frosch/test/Overlap/ShyLU_DDFROSch_overlap.exe"
parameter_path = "/faststorage/ianscluster08/kaiserls/Trilinos/packages/shylu/shylu_dd/frosch/test/Overlap/ParameterLists/"
hostfile_path = "/faststorage/ianscluster08/kaiserls/Trilinos/Hostfiles/"

def getParameterFile(mode):
    return parameter_path + f"ParameterList_OneLevelPreconditioner_{mode}.xml"
def getHostFile(N):
    return hostfile_path + f"hostfile_{N}.txt"
def getOutFile(N,M,O,mode):
    outfile = f"output_N{N}_M{M}_O{O}_{mode}.txt"
    return outfile
def getExecutableFile():
    return executable_path

def getCommandString(N,M,O,mode):
    mpi = f"nice -n 20 mpirun -n {N} -mca btl self,openib,vader,tcp -mca btl_tcp_if_exclude lo,docker0,wan1"
    host = f"--hostfile {getHostFile(N)}"
    exe = getExecutableFile()
    opts = f"--M={M} --DIM=2 --O={O} --DPN=1"
    plist = f"--PLIST={getParameterFile(mode)}"
    return " ".join([mpi, host, exe, opts, plist])

def cleanup(output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    if not os.path.exists(output_path):
            os.makedirs(output_path)

def compute(output_path, Ms, Ns, Os, Modes, max_fails=10):
    for M in Ms:
        for N in Ns:
            for O in Os:
                for mode in Modes:
                    res = 1
                    fails=0
                    while res!=0 and fails<max_fails:
                        try:
                            with open(output_path+getOutFile(N,M,O,mode), "w") as outfile:
                                cla = shlex.split(getCommandString(N,M,O,mode))
                                #cla = shlex.split("echo 1")
                                res = subprocess.call(cla,cwd=output_path,stdout=outfile)
                                if(res!=0): #failed
                                    fails=fails+1
                                    raise Exception()   
                        except:
                            print(f"failed for {output_path+getOutFile(N,M,O,mode)}")
                            os.remove(output_path+getOutFile(N,M,O,mode))
                        yield
    print("---------------FINISHED---------------")

from alive_progress import alive_bar

def main():
    output_path = os.getcwd()+'/'+'output/'
    cleanup(output_path)

    Ms = [200,400,800]
    Ns = [4,16,64,144]
    Os = [8,12,20]
    Modes = ["cg", "asho", "asho_onOverlapping", "rasho"]
    computer = compute(output_path, Ms, Ns, Os, Modes)

    n = len(Ns)*len(Ms)*len(Os)*len(Modes)
    print(f"Number of combinations, which will be running: {n}")


    with alive_bar(n) as bar:
        for sample in computer:
            bar()
    
if __name__=="__main__":
    main()