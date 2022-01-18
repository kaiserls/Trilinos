//@HEADER
// ************************************************************************
//
//               ShyLU: Hybrid preconditioner package
//                 Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Alexander Heinlein (alexander.heinlein@uni-koeln.de)
//
// ************************************************************************
//@HEADER

#ifndef _FROSCH_DEBUGGING_HPP
#define _FROSCH_DEBUGGING_HPP

#include <fstream>

namespace FROSch{
    using namespace std;


inline void writeMeta(int nx, int ny, int processes){
    ofstream myfile;
    myfile.open ("meta.txt");
    myfile << nx << " "<<ny <<" " <<processes;
    myfile.close();
}

template <class vector_type, class map_type>
inline void output(const vector_type vec, const map_type globalMap ){
    auto map = vec->getMap();
    int proc= map ->getComm()->getRank();
    // Write meta once
    if(proc == 0){
        int n = int(sqrt(globalMap->getGlobalNumElements()));//unique map
        int procs = map ->getComm()->getSize();
        writeMeta(n,n, procs);
    }
    ofstream myfile;
    myfile.open ("nodes"+std::to_string(proc)+".txt");
    for(auto node: map->getNodeElementList()){
        myfile<<node<<" ";
    }
    myfile.close();
    myfile.open ("values"+std::to_string(proc)+".txt");
    auto values = vec->getData(0);
    for (auto j=0; j<map->getNodeNumElements(); j++) {
        myfile<<values[j]<<" ";
    }
    myfile.close();
}

}
#endif