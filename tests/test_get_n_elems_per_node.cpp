#include <mpi.h>
#include <iostream>
#include "../include/mpiworker/mpiworker.hpp"

#define BOOST_TEST_MODULE test_get_n_elems_per_node
#include <boost/test/included/unit_test_framework.hpp>

#define DEBUG 0

/*! \russian Выполняется тестирование метода getNElemsPerNode. Для сборки только этого теста выполните команду \code make make test_get_n_elems_per_node \endcode  и запустите его на исполнение, например \code mpirun -np 3 ./test_get_n_elems_per_node \endcode
 */
BOOST_AUTO_TEST_CASE( test_get_n_elems_per_node )
{
    int N = 7;

    mpiworker::MPIWorker a;

    if( a.getNNodes() != 3 )
    {
        std::cerr << "Run the command: mpirun -np 3 ./test_get_n_elems_per_node\n";
        MPI::COMM_WORLD.Abort( MPI_ERR_ARG );
    }

    a.setMode(0);
    a.setNElems(N);

#if DEBUG > 0
    if( a.getRankNode()==0 )  a.print();   MPI::COMM_WORLD.Barrier();
    if( a.getRankNode()==1 )  a.print();   MPI::COMM_WORLD.Barrier();
    if( a.getRankNode()==2 )  a.print();   MPI::COMM_WORLD.Barrier();
#endif
    
    if( a.getRankNode() == 0) 
    {
        BOOST_CHECK( a.getNElemsPerNode() == 0 );
    }

    if( a.getRankNode() == 1) 
    {
        BOOST_CHECK( a.getNElemsPerNode() == 3 );
    }

    if( a.getRankNode() == 2) 
    {
        BOOST_CHECK( a.getNElemsPerNode() == 4 );
    }

    a.setMode(1);

#if DEBUG > 0
    if( a.getRankNode()==0 )  a.print();   MPI::COMM_WORLD.Barrier();
    if( a.getRankNode()==1 )  a.print();   MPI::COMM_WORLD.Barrier();
    if( a.getRankNode()==2 )  a.print();   MPI::COMM_WORLD.Barrier();
#endif

    if( a.getRankNode() == 0) 
    {
        BOOST_CHECK( a.getNElemsPerNode() == 2 );
    }

    if( a.getRankNode() == 1) 
    {
        BOOST_CHECK( a.getNElemsPerNode() == 2 );
    }

    if( a.getRankNode() == 2) 
    {
        BOOST_CHECK( a.getNElemsPerNode() == 3 );
    }

    a.setMode(1);
}
