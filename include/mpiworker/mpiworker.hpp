/** @addtogroup MPIWorker
* @{*/

    /** @file */


/*!
* \mainpage Описание проекта
* 
* ### Класс mpiworker::MPIWorker
* 
* Содержит методы-обертки с простым синтаксисом для коллективных операций [MPI](https://www.open-mpi.org/) (Scatterv, Gatherv и других). 
* Данные храняться в векторах [std::vector](http://ru.cppreference.com/w/cpp/container/vector), работа выполняется в коммуникаторе MPI::COMM_WORLD.
*
* Для выполнения коллективных операций требуется создать объект класса MPIWorker:
* \code
* mpiworker::MPIWorker w;
* \endcode
* затем установить режим работы и общее число обрабатываемых элементов
* \code
* w.setMode(0);
* w.setNElems(9999);
* \endcode
*
* Пример разделения элементов вектора на приблизительно равные части:
* \code                                               // result for nNodes = 3
*                                                     // that is mpirun -np 3 ./example_scatter_then_gather
*
*    mpiworker::MPIWorker w;                          // MPI::Init(), Get_size() and Get_rank()
*
*    int N = 0; 
*    std::vector<float> x; 
*
*    if( !w.getRankNode() ) 
*    { 
*        N = 11; 
*        x.resize(N);
*        std::iota(x.begin(),x.end(),1);              // rank=0: x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
*    }
*
*    w.bcast<int>(N,MPI::INT);                        // N == 11 on all ranks
*    w.setMode(1);                                    // all nodes have equal rights
*    w.setNElems(N);                                  // counts = { 3, 4, 4 }
*                                                     // displacement  = { 0, 3, 7 }
*    if( !w.getRankNode() ) w.print();
*
*    std::vector<float> xPerNode;                     // vector for local portions
*    w.scatterv<float>(x,xPerNode,MPI::FLOAT);        // rank=0: { 1, 2, 3 }
*                                                     // rank=1: { 4, 5, 6, 7 }
*                                                     // rank=2: { 8, 9, 10, 11 }
*    for( auto & e: xPerNode)
*    {
*        e += w.getRankNode();                        // rank=0: { 1, 2, 3 }
*    }                                                // rank=1: { 5, 6, 7, 8 }
*                                                     // rank=2: { 10, 11, 12, 13 }
*    std::vector<float> y( N );
*    w.gatherv( xPerNode, y, MPI::FLOAT );            // rank=0: { 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13 }
*                                                     // rank=1,2: { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
*
*    w.allGatherv( xPerNode, y, MPI::FLOAT );         // rank=0,1,2: { 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13 }
* \endcode
* 
*
* Класс mpiworker::MPIInit и функция  calculatePortions играют вспомогательную роль, но могут быть использованы независимо от mpiworker::MPIWorker.
* 
* ### Класс mpiworker::MPIInit
* 
* Выполняет инициализацию и закрытие библиотеки MPI. Реализован как Singleton Meyers. 
* Благодаря этому можно организовывать и хранить произвольное число схем разбиения массивов в программе:
* \code
*                                                    // mpirun -np 3 ./example_two_mpiworkers
*    std::vector<float> x1, x1PerNode,
*                       x2, x2PerNode;
*    int N1, N2;
*
*    mpiworker::MPIWorker w1;                
*
*    mpiworker::MPIWorker w2;
*
*    if( !w1.getRankNode() ) 
*    {
*        x1 = { 1, 2, 3, 4, 5 }; 
*        N1 = x1.size();
*        x2 = { 6, 7, 8, 9, 10, 11, 12};
*        N2 = x2.size();
*    }
*
*
*    w1.bcast<int>(N1,MPI::INT);
*    w1.bcast<int>(N2,MPI::INT);
*
*    w1.setMode(0);                                  // counts = { 0, 2, 3 }
*    w1.setNElems(N1);                               // displs = { 0, 0, 2 }
*
*    w2.setMode(0);                                  // counts = { 0, 3, 4 }
*    w2.setNElems(N2);                               // displs = { 0, 0, 3 }
*
*    w1.scatterv<float>(x1,x1PerNode,MPI::FLOAT);    // rank=0: {}
*                                                    // rank=1: { 1, 2 }
*                                                    // rank=2: { 3, 4, 5 }
*    w2.scatterv<float>(x2,x2PerNode,MPI::FLOAT);    // rank=0: {}
*                                                    // rank=1: { 6, 7, 8 }
*                                                    // rank=2: { 9, 10, 11, 12 }
* \endcode
*
* ### Функция [calculatePortions](group__MPIWorker.html#ga6fd8303c1b4e39a4a623756fdcbeae6f) 
*
* Выполняет формирование вспомогательных массивов для деления некоторого общего количества элементов на приблизительно равные части коллективной 
* операцией Scatterv или для объединения массивов значений на узлах в единый массив.
*/

#ifndef CLASS_MPI_WORKER_NDN_2016
#define CLASS_MPI_WORKER_NDN_2016

#include <vector>
#include <iterator>
#include <algorithm>

#include "tools_for_parallel.hpp"

namespace mpiworker
{
    
    /*! \brief \~russian Создает и удаляет коммуникатор MPI::COMM_WORLD.
     *  \~russian Реализован как Singleton Meyers. Начиная с С++11 потокобезобасен [ISO N3337, 6.7.4].
     */ 
    class MPIInit{
    
        //! \~russian Число узлов.
        int nNodes_;
    
        //! \~russian Ранг узла.
        int rankNode_;
    
        //! \~russian Конструктор.
        MPIInit()
        {
            MPI::Init();
            nNodes_ = MPI::COMM_WORLD.Get_size(); 
            rankNode_ = MPI::COMM_WORLD.Get_rank(); 
        }
    
        //! \~russian Деструктор.
        ~MPIInit(){ MPI::Finalize(); } 
    public:
    
        //! \~russian 
        static MPIInit& instance()
        {
            static MPIInit comm;
            return comm;
        }
    
        //! Возвращает ранг узла.
        int getRankNode() const { return rankNode_; }
    
        //! Возвращает число узлов.
        int getNNodes() const  { return nNodes_; }
    };
    
    
    /*! \brief \~russian Содержит методы, выполняющие разделение и сборку массивов в MPI-приложении.
     *
     * \~russian Прежде всего представляет  собой обертку к коллективным MPI-функциям типа Scatterv и Gatherv. Работа выполняется в коммуникаторе COMM_WORLD. 
     * Реализованы схемы:  
     * - с управляющим нулевым узлом при setMode(0); 
     * - все узлы выполняют вычисления при setMode(1).
     *
     *   \~russian Предлагает упрощенный синтаксис вызова некоторых коллективных операций из стандарта MPI.
     */
    class MPIWorker
    {
        //! \~russian Ссылка на 
        MPIInit& comm = MPIInit::instance();
    
        //! \~russian Общее число обрабатываемых элементов. \details \~russian Длина разрезаемого или собираемого массива.
        int nElems_ { 0 };

        //! \~russian Число элементов для  текущего узла. \details \~russian Длина массива для текущего узла, ранг которого определен переменной rankNode_.
        int nElemsPerNode_ { 0 };
    
        //! \~russian Режим работы узлов кластера. \detail \~russian Доступно 0 (нулевой узел управляющий) или 1 (все узлы выполняют вычисления).
        short mode_ { 0 };
    
        //! \~russsian Ранг узла.
        int rankNode_ { 0 };
    
        //! \~russian Число узлов.
        int nNodes_ { 0 };
    
        //! \~russian Число элементов для каждого из узлов. \details \~russian Формат соответствует формату, принятому в MPI для коллективных операций типа Scatterv и Gatherv.
        std::vector<int> countsElemsPerNode_ { };
    
        //! \~russian Смещения элементов в общем массиве для каждого из узлов.
        std::vector<int> displsElemsPerNode_ { };

        //! \~russian Выполняет расчет нагрузки для узлов.
        void calculate()
        {
            if( !rankNode_ )
            {
                calculatePortions
                (
                    nElems_,
                    countsElemsPerNode_.begin(),
                    countsElemsPerNode_.end(),
                    displsElemsPerNode_.begin(),
                    displsElemsPerNode_.end(),
                    mode_
                );
            }
    
            MPI::COMM_WORLD.Bcast
            ( 
                countsElemsPerNode_.data(), 
                nNodes_, 
                MPI::INT, 
                0 
            );
    
            MPI::COMM_WORLD.Bcast
            ( 
                displsElemsPerNode_.data(), 
                nNodes_, 
                MPI::INT, 
                0 
            );
    
            nElemsPerNode_ = countsElemsPerNode_[rankNode_];
        }
    
    public:
    
        //! \~russian Конструктор.
        MPIWorker() : rankNode_( comm.getRankNode() ), nNodes_( comm.getNNodes() )
        {
            countsElemsPerNode_.resize( nNodes_ );
            displsElemsPerNode_.resize( nNodes_ );
        }
    
    
        /*! \~russian Устанавливает режим работы узлов кластера. \details 
         * \~russian Доступные режимы:
         * \~russian * 0 - управляющий нулевой узел 
         * \~russian * 1 - все узлы равноправны
         */
        void setMode( const short & mode ) 
        { 
            mode_ = mode; 
            calculate();
        }
    
        //! \~russian Число элементов в разрезаемом или собираемом массиве.
        void setNElems(const int & nElems)
        {
            nElems_ = nElems;
            calculate();
        }
    
    
    
        //! \~russian Возвращает ранг узла.
        int getRankNode() const { return rankNode_; }
    
        //! \~russian Возвращает число узлов.
        int getNNodes() const { return nNodes_; }
   
        //! \~russian Возвращает число элементов, которые должны быть обработаны на текущем узле
        int getNElemsPerNode() const { return nElemsPerNode_; }
    
        //! \~russian Разделение элементов массива на приблизительно равные части. \details \~russian \param[in] array Исходный массив со всеми элементами. \param[out] arrayPerNode Выходной массив с элементами для текущего узла. \param[in] MPIType Тип элементов. 
        template <typename T>
        void scatterv( const std::vector<T> & array, std::vector<T> & arrayPerNode,  MPI::Datatype MPIType ) 
        {
            if( arrayPerNode.size() != nElemsPerNode_ ) arrayPerNode.resize( nElemsPerNode_ );
    
            MPI::COMM_WORLD.Scatterv
            (
                array.data(), 
                countsElemsPerNode_.data(), 
                displsElemsPerNode_.data(), 
                MPIType, 
                arrayPerNode.data(), 
                nElemsPerNode_, 
                MPIType, 
                0 
            );
        }
    
    
    
        //! \~russian Сбор элементов в единые массивы на всех узлах. \details \~russian \param[in] arrayPerNode Входной массив с элементами текущего узла.  \param[out] array Итоговый массив со всеми элементами. \param[in] MPIType Тип элементов.
        template <typename T>
        void allGatherv( const std::vector<T> & arrayPerNode, std::vector<T> & array,  MPI::Datatype MPIType )
        {
            if( array.size() != nElems_ ) array.resize( nElems_ );

            MPI::COMM_WORLD.Allgatherv
            (
                arrayPerNode.data(),
                nElemsPerNode_,
                MPIType,
                array.data(),
                countsElemsPerNode_.data(),
                displsElemsPerNode_.data(),
                MPIType
            );
        }
    
    
        //! \~russian Сбор элементов в единый массив на нулевом узле. \details \~russian \param[in] arrayPerNode Входной массив с элементами текущего узла.  \param[out] array Итоговый массив со всеми элементами. \param[in] MPIType Тип элементов.
        template <typename T>
        void gatherv( const std::vector<T> & arrayPerNode, std::vector<T> & array,  MPI::Datatype MPIType )
        {
            if( array.size() != nElems_ && !rankNode_ ) array.resize( nElems_ );

            MPI::COMM_WORLD.Gatherv
            (
                arrayPerNode.data(),
                nElemsPerNode_,
                MPIType,
                array.data(),
                countsElemsPerNode_.data(),
                displsElemsPerNode_.data(),
                MPIType,
                0
            );
        }
    
        //! \~russian Рассылает значение скалярной переменной с нулевого узла на все остальные.
        template <typename T>void bcast( T & var, MPI::Datatype MPIType )
        {
            MPI::COMM_WORLD.Bcast( &var, 1, MPIType, 0 );
        }
    
        //! \~russian Выполняет редукцию со сбором результата на нулевом узле
        template <typename T>void reduce( const std::vector<T> & arrayPart, std::vector<T> & arrayRes, MPI::Datatype MPIType, MPI::Op MPIOp )
        {
            if( arrayRes.size() != nElems_ && !rankNode_ ) arrayRes.resize( nElems_ );

            MPI::COMM_WORLD.Reduce
            ( 
                arrayPart.data(), 
                arrayRes.data(), 
                arrayPart.size(), 
                MPIType, 
                MPIOp, 
                0 
            );
        }
    
        //! \~russian Выполняет редукцию с сохранением результата на всех узлах
        template <typename T>void allReduce( const std::vector<T> & arrayPart, std::vector<T> & arrayRes, MPI::Datatype MPIType, MPI::Op MPIOp )
        {
            if( arrayRes.size() != nElems_ ) arrayRes.resize( nElems_ );

            MPI::COMM_WORLD.Allreduce
            ( 
                arrayPart.data(), 
                arrayRes.data(), 
                arrayRes.size(), 
                MPIType, 
                MPIOp 
            );
        }
    
        //! \~russian Печать значений, хранимых в полях.
        void print()
        {
            std::cout << "\033[34;4mMPIWorker debug:\033[0m\n";
    
            std::cout << "\033[34;1m    nNodes_\033[0;36;2m = " << nNodes_ << "\033[0m" << std::endl;
    
            std::cout << "\033[34;1m    rankNode_\033[0;36;2m = " << rankNode_ << "\033[0m" << std::endl;
    
            std::cout << "\033[34;1m    nElems_\033[0;36;2m = " << nElems_ << "\033[0m" << std::endl;
    
            std::cout << "\033[34;1m    mode_\033[0;36;2m = " << mode_ << "\033[0m" << std::endl;

            std::cout << "\033[34;1m    nElemsPerNode_\033[0;36;2m = " << nElemsPerNode_ << "\033[0m" << std::endl;
    
            std::cout << "\033[34;1m    countsElemsPerNode_\033[0;36;2m = ";
            std::copy
            (
                countsElemsPerNode_.begin(),
                countsElemsPerNode_.end(),
                std::ostream_iterator<int>( std::cout, " " )
            );
            std::cout << "\033[0m" << std::endl;
    
            std::cout << "\033[34;1m    displsElemsPerNode_\033[0;36;2m = ";
            std::copy
            (
                displsElemsPerNode_.begin(),
                displsElemsPerNode_.end(),
                std::ostream_iterator<int>( std::cout, " " )
            );
            std::cout << "\033[0m" << std::endl;
        }
    
    
        //! \~russian Деструктор.
        ~MPIWorker(){}
    };
    
    
} // namespace mpiworker
    
#endif

/*@}*/
