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

#ifndef _FROSCH_OVERLAPPINGOPERATOR_DECL_HPP
#define _FROSCH_OVERLAPPINGOPERATOR_DECL_HPP

#include <FROSch_SchwarzOperator_def.hpp>
#include <FROSch_Mapper_def.hpp>


namespace FROSch {

    using namespace Teuchos;
    using namespace Xpetra;

    //! A SchwarzOperator which belongs to an overlapping domain decomposition.
    //! This allows to implement the apply operation of the SchwarzOperator.
    template <class SC = double,
              class LO = int,
              class GO = DefaultGlobalOrdinal,
              class NO = KokkosClassic::DefaultNode::DefaultNodeType>
    class OverlappingOperator : public SchwarzOperator<SC,LO,GO,NO> {

    protected:

        using CommPtr               = typename SchwarzOperator<SC,LO,GO,NO>::CommPtr;

        using XMapPtr               = typename SchwarzOperator<SC,LO,GO,NO>::XMapPtr;
        using ConstXMapPtr          = typename SchwarzOperator<SC,LO,GO,NO>::ConstXMapPtr;

        using XMatrixPtr            = typename SchwarzOperator<SC,LO,GO,NO>::XMatrixPtr;
        using ConstXMatrixPtr       = typename SchwarzOperator<SC,LO,GO,NO>::ConstXMatrixPtr;

        using XMultiVector          = typename SchwarzOperator<SC,LO,GO,NO>::XMultiVector;
        using XMultiVectorPtr       = typename SchwarzOperator<SC,LO,GO,NO>::XMultiVectorPtr;

        using XImportPtr            = typename SchwarzOperator<SC,LO,GO,NO>::XImportPtr;
        using XExportPtr            = typename SchwarzOperator<SC,LO,GO,NO>::XExportPtr;

        using ParameterListPtr      = typename SchwarzOperator<SC,LO,GO,NO>::ParameterListPtr;

        using SolverPtr             = typename SchwarzOperator<SC,LO,GO,NO>::SolverPtr;
        using SolverFactoryPtr      = typename SchwarzOperator<SC,LO,GO,NO>::SolverFactoryPtr;

        using SCVecPtr              = typename SchwarzOperator<SC,LO,GO,NO>::SCVecPtr;
        using ConstSCVecPtr         = typename SchwarzOperator<SC,LO,GO,NO>::ConstSCVecPtr;

        using UN                    = typename SchwarzOperator<SC,LO,GO,NO>::UN;
        
        using MapperPtr             = RCP<Mapper<SC,LO,GO,NO>>;
        using OverlapCombinationType= typename Mapper<SC,LO,GO,NO>::OverlapCombinationType;

    public:
        using SchwarzOperator<SC,LO,GO,NO>::apply;

        OverlappingOperator(ConstXMatrixPtr k,
                            ParameterListPtr parameterList);

        ~OverlappingOperator();

        virtual int initialize() = 0;

        virtual int compute() = 0;

        virtual void apply(const XMultiVector &x,
                           XMultiVector &y,
                           bool usePreconditionerOnly,
                           ETransp mode=NO_TRANS,
                           SC alpha=ScalarTraits<SC>::one(),
                           SC beta=ScalarTraits<SC>::zero()) const;

    protected:
        virtual int initializeOverlappingOperator();

        virtual int initializeSubdomainSolver(ConstXMatrixPtr localMat);

        virtual int computeOverlappingOperator();

        virtual int updateLocalOverlappingMatrices();

        virtual void restrictFromInto(const XMultiVectorPtr source, XMultiVectorPtr & target) const;
        virtual void prolongateFromInto(const XMultiVectorPtr source, XMultiVectorPtr target, const ConstXMapPtr uniqueMap) const;
        int calculateMultiplicity();

        //TODO: Rename. Code would be much easier to understand if "LocalOverlappingMatrix_" and GlobalOverlappingMap_"
        ConstXMatrixPtr OverlappingMatrix_; //! Local overlapping matrix (neglecting the initialization with the globally distributed matrix K_)
        ConstXMapPtr OverlappingMap_; //! Distribution of the nodes/node indices over the ranks

        // Temp Vectors for apply()
        mutable XMultiVectorPtr XTmp_;
        mutable XMultiVectorPtr XOverlap_;
        mutable XMultiVectorPtr XOverlapTmp_;
        mutable XMultiVectorPtr YOverlap_;

        XImportPtr Scatter_; //! TODO: Comment: Describes how to exchange data between the overlapping map and the global uniquely distributed map???

        SolverPtr SubdomainSolver_; //! Solver for the local problem on this subdomain. Used each time the operator is applied.

        XMultiVectorPtr Multiplicity_; //! Stores in how many domains each node is contained.

        OverlapCombinationType Combine_ = OverlapCombinationType::Averaging;

        MapperPtr Mapper_;
    };

}

#endif
