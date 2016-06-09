#include <mpi.h>
#include <iostream>
#include <algorithm>
#

#include "../include/mpiworker/mpiworker.hpp"


int main(){
                                                    // mpirun -np 3 ./example_two_mpiworkers
    std::vector<float> x1, x1PerNode,
                       x2, x2PerNode;
    int N1, N2;

    mpiworker::MPIWorker w1;                

    mpiworker::MPIWorker w2;

    if( !w1.getRankNode() ) 
    {
        x1 = { 1, 2, 3, 4, 5 }; 
        N1 = x1.size();
        x2 = { 6, 7, 8, 9, 10, 11, 12};
        N2 = x2.size();
    }


    w1.bcast<int>(N1,MPI::INT);
    w1.bcast<int>(N2,MPI::INT);

    w1.setMode(0);                                  // counts = { 0, 2, 3 }
    w1.setNElems(N1);                               // displs = { 0, 0, 2 }

    w2.setMode(0);                                  // counts = { 0, 3, 4 }
    w2.setNElems(N2);                               // displs = { 0, 0, 3 }

    w1.scatterv<float>(x1,x1PerNode,MPI::FLOAT);    // rank=0: {}
                                                    // rank=1: { 1, 2 }
                                                    // rank=2: { 3, 4, 5 }
    w2.scatterv<float>(x2,x2PerNode,MPI::FLOAT);    // rank=0: {}
                                                    // rank=1: { 6, 7, 8 }
                                                    // rank=2: { 9, 10, 11, 12 }
                                                    
    // print:
    
    if( !w1.getRankNode() ) { w1.print(); w2.print(); }

    std::cout << "rank: " << w1.getRankNode() << " x1PerNode: " ;
    std::copy(x1PerNode.begin(),x1PerNode.end(),std::ostream_iterator<float>(std::cout, " " ));
    std::cout << std::endl;

    std::cout <<  " x2PerNode: " ;
    std::copy(x2PerNode.begin(),x2PerNode.end(),std::ostream_iterator<float>(std::cout, " " ));
    std::cout << std::endl;


    return 0;
}
