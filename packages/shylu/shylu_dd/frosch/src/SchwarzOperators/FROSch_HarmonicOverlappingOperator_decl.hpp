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

#ifndef _FROSCH_HARMONICOVERLAPPINGOPERATOR_DECL_HPP
#define _FROSCH_HARMONICOVERLAPPINGOPERATOR_DECL_HPP

#include <FROSch_AlgebraicOverlappingOperator_def.hpp>

//TODO: Use get export lids for determining which entries will be sent -> are multiple/interface/....?
// or findUnionTargetGIDs 
namespace FROSch {

    using namespace std;
    using namespace Teuchos;
    using namespace Xpetra;

    /**
     * @brief The PreSolveStrategy decides on what domain the preSolve step is performed and the right hand side is set to zero.
     * 
     */
    enum PreSolveStrategy {
        OnOvlp=0,       /// PreSolve on the ovlp domain which consist of small disconnected node sets -> Solve is cheap
        OnOverlapping=1 /// PreSolve on the domain which is also used for apply()-ing the operator -> Setup of preSolve is for free
    };


    /**
     * @brief This class implements an overlapping operator which enforces and exploits a harmonic decay of the solution on the overlap.
     * It follows the papers:
     * - Additive Version: "Sarkis, Marcus. "Partition of unity coarse spaces and Schwarz methods with harmonic overlap." Recent Developments
     * in Domain Decomposition Methods. Springer, Berlin, Heidelberg, 2002. 77-94."
     * - Restricted Version: X.-C. Cai, M. Dryja, and M. Sarkis, “Restricted additive schwarz preconditioners with harmonic overlap for symmetric positive definite
     * linear systems,” SIAM Journal on Numerical Analysis, vol. 41, no. 4, pp. 1209–1231, 2003.
     */
    template <class SC = double,
              class LO = int,
              class GO = DefaultGlobalOrdinal,
              class NO = KokkosClassic::DefaultNode::DefaultNodeType>
    class HarmonicOverlappingOperator : public AlgebraicOverlappingOperator<SC,LO,GO,NO> {

    protected:

        using CommPtr               = typename SchwarzOperator<SC,LO,GO,NO>::CommPtr;

        using XMapPtr               = typename SchwarzOperator<SC,LO,GO,NO>::XMapPtr;
        using ConstXMapPtr          = typename SchwarzOperator<SC,LO,GO,NO>::ConstXMapPtr;

        using XMatrixPtr            = typename SchwarzOperator<SC,LO,GO,NO>::XMatrixPtr;
        using ConstXMatrixPtr       = typename SchwarzOperator<SC,LO,GO,NO>::ConstXMatrixPtr;

        using XMultiVector          = typename SchwarzOperator<SC,LO,GO,NO>::XMultiVector;
        using XMultiVectorPtr       = typename SchwarzOperator<SC,LO,GO,NO>::XMultiVectorPtr;

        using ConstXCrsGraphPtr     = typename SchwarzOperator<SC,LO,GO,NO>::ConstXCrsGraphPtr;

        using ParameterListPtr      = typename SchwarzOperator<SC,LO,GO,NO>::ParameterListPtr;
        
        using SolverPtr             = typename SchwarzOperator<SC,LO,GO,NO>::SolverPtr;
        using SolverFactoryPtr      = typename SchwarzOperator<SC,LO,GO,NO>::SolverFactoryPtr;

        using MapperPtr             = RCP<Mapper<SC,LO,GO,NO>>;
        using CombinationType       = typename Mapper<SC,LO,GO,NO>::CombinationType;

    public:

        HarmonicOverlappingOperator(ConstXMatrixPtr k,
                                     ParameterListPtr parameterList);

        virtual int initializeOverlappingOperator();

        virtual int compute();
        virtual string description() const;

        virtual void apply(const XMultiVector &x,
                           XMultiVector &y,
                           bool usePreconditionerOnly,
                           ETransp mode=NO_TRANS,
                           SC alpha=ScalarTraits<SC>::one(),
                           SC beta=ScalarTraits<SC>::zero()) const;

        virtual void preSolve(XMultiVector & rhs);
        virtual void afterSolve(XMultiVector & lhs);

    protected:

        bool HarmonicOnOverlap_ = false; //! Use harmonic decay of subdomain "solution" on overlap
        bool Rasho_ = false; //Use the restricted mode of the preconditioner
                            // Be carefull, this->Combine_ is resetted to additive internally after the constructor if using Rasho_=True;
        PreSolveStrategy PreSolveStrategy_= PreSolveStrategy::OnOvlp;
        
        ConstXMapPtr PreSolveMap_; //Contains the nodes where the system matrix should be imported and the system solved during presolve
        ConstXMapPtr InnerMap_; //Contains the nodes on which the residum needs to be imported
        ConstXMapPtr LocalSolveMap_; //Contains the nodes where the system matrix should be imported and the system solved during apply

        MapperPtr PreSolveMapper_; //Mapper used for pre-/afterSolve
        MapperPtr InnerMapper_; //Mapper used for import in harmonic apply

        // TODO: Remove intermediate step for performance reasons?
        //Mappers used to import the residual on the inner nodes and extend with zero to the extended domain.
        MapperPtr UniqueToInnerMapper_;
        MapperPtr InnerToOverlappingMapper_;
        mutable XMultiVectorPtr IntermediateInner_;

        XMultiVectorPtr W_;
        mutable XMultiVectorPtr RhsPreSolveTmp_;
        SolverPtr HarmonicSolver_;
    
    private:
        virtual int setupHarmonicSolver();
        virtual int calculateHarmonicMapsByMultiplicity(RCP<const CrsGraph<LO,GO,NO> > graph, RCP<MultiVector<SC,LO,GO,NO>> multiplicity);
    };

}

#endif
