#include <mpi.h>
#include <iostream>
#include <algorithm>
#

#include "../include/mpiworker/mpiworker.hpp"


int main(){


  mpiworker::MPIWorker w;                        // MPI::Init(), Get_size() and Get_rank()

    int N = 0; 
    std::vector<float> x; 

    if( !w.getRankNode() ) 
    { 
        N = 11; 
        x.resize(N);
        std::iota(x.begin(),x.end(),1);             // rank=0: x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
    }

    w.bcast<int>(N,MPI::INT);                       // N == 11 on all ranks
    w.setMode(1);                                   // all nodes have equal rights
    w.setNElems(N);                                 // counts = { 3, 4, 4 }
                                                    // displacement  = { 0, 3, 7 }
    if( !w.getRankNode() ) w.print();

    std::vector<float> xPerNode;                     // vector for local portions
    w.scatterv<float>(x,xPerNode,MPI::FLOAT);        // rank=0: { 1, 2, 3 }
                                                     // rank=1: { 4, 5, 6, 7 }
                                                     // rank=2: { 8, 9, 10, 11 }
    for( auto & e: xPerNode)
    {
        e += w.getRankNode();                        // rank=0: { 1, 2, 3 }
    }                                                // rank=1: { 5, 6, 7, 8 }
                                                     // rank=2: { 10, 11, 12, 13 }
    std::vector<float> y( N );
    w.gatherv( xPerNode, y, MPI::FLOAT );            // rank=0: { 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13 }
                                                     // rank=1,2: { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }

    w.allGatherv( xPerNode, y, MPI::FLOAT );         // rank=0,1,2: { 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13 }

    // print:

    std::cout << "rank: " << w.getRankNode() << " xPerNode: " ;
    std::copy(xPerNode.begin(),xPerNode.end(),std::ostream_iterator<float>(std::cout, " " ));

    std::cout  << " y: " ;
    std::copy(y.begin(),y.end(),std::ostream_iterator<float>(std::cout, " " ));
    std::cout << std::endl;


    return 0;
}
