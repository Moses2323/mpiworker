#include <mpi.h>
#include <iostream>
#include "../include/mpiworker/mpiworker.hpp"

#define BOOST_TEST_MODULE test_scatter_and_gather
#include <boost/test/included/unit_test_framework.hpp>

/*! \russian Выполняется тестирование операторов scattrv и gatherv. Для сборки только этого теста выполните команду \code make test_scatter_then_gather \endcode  и запустите его на исполнение, например \code mpirun -np 2 ./test_scatter_then_gather \endcode
 */
BOOST_AUTO_TEST_CASE( test_scatter_and_gather )
{

    std::cout << "test for scatterv and gatherv\n";

    std::vector<int> x;
    std::vector<int> xPerNode;
    int N;

    mpiworker::MPIWorker a;

    if( !a.getRankNode() ) 
    {
        x.resize(11);
        x = {1,2,3,4,5,6,7,8,9,10,11};
        N = x.size();
    }

    a.setMode(1);
    a.bcast(N,MPI::INT);
    a.setNElems(N);

    a.scatterv<int>(x,xPerNode,MPI::INT);

    std::vector<int> y;

    if( !a.getRankNode() ) 
    {
        y.resize(x.size());
    }

    a.gatherv<int>(xPerNode,y,MPI::INT);

    
    if( !a.getRankNode() ) 
    {
        BOOST_CHECK_EQUAL_COLLECTIONS(x.begin(),x.end(),y.begin(),y.end());
    }
}
