/** @addtogroup MPIWorker
 * @{*/

 /** @file */


#include <algorithm>


/*! 
 * \brief \~russian Функция выполняет разделение элементов на приблизительно одинаковые порции.
 * \~russian Формирует вспомагательные массивы для MPI функций  MPI_Gatherv and MPI_Scatterv. Поддерживает схемы: 
 *  - с управляющим нулевым узлом;
 *  - все узлы равноправны.
 */
template <typename Iterator, typename SizeType>
void calculatePortions(

    //! \~russian Число элементов
    SizeType nElems,

    // range for the container with portions
    Iterator begCounts,  
    Iterator endCounts,

    // range for the container with offset
    Iterator begDispls,
    Iterator endDispls,

    // isZeorWork == 0 --- zero-node is manager
    bool  isZeroWork = false 

)
{
    typename std::iterator_traits<Iterator>::difference_type nNodes = std::distance( begCounts, endCounts );
    
    if( !isZeroWork ) // zero node --- manager
    {
        if( nNodes <= 1 ) return;

        SizeType main = nElems / ( nNodes - 1 );
        SizeType balance = nElems % ( nNodes - 1 );
        *begCounts = *begDispls = 0;
        ++begCounts;
        ++begDispls;
        while( begCounts != endCounts ){
            *begCounts = std::distance( begCounts, endCounts )<=balance ? main+1 : main;
            *begDispls = *(begDispls-1) + *(begCounts-1);
            ++begDispls;
            ++begCounts;
        }

    } 
    else // no any managers
    {
        SizeType main = nElems / nNodes ;
        SizeType balance = nElems % nNodes ;
        while( begCounts != endCounts ){
            *begCounts = std::distance( begCounts, endCounts )<=balance ? main+1 : main;
            *begDispls = *(begDispls-1) + *(begCounts-1);
            ++begDispls;
            ++begCounts;
        }
    }
}

/*@}*/
