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
// Questions? Contact Christian Hochmuth (c.hochmuth@uni-koeln.de)
//
// ************************************************************************
//@HEADER

#ifndef _FROSCH_MULTIPLICATIVEOPERATOR_DECL_HPP
#define _FROSCH_MULTIPLICATIVEOPERATOR_DECL_HPP

#include <FROSch_ComposedOperator_def.hpp>


namespace FROSch {

    using namespace std;
    using namespace Teuchos;
    using namespace Xpetra;

    //! Multiplicative composition of SchwarzOperators
    template <class SC = double,
              class LO = int,
              class GO = DefaultGlobalOrdinal,
              class NO = KokkosClassic::DefaultNode::DefaultNodeType>
    class MultiplicativeOperator : public ComposedOperator<SC,LO,GO,NO> {

    protected:

        using CommPtr                   = typename ComposedOperator<SC,LO,GO,NO>::CommPtr;

        using XMapPtr                   = typename ComposedOperator<SC,LO,GO,NO>::XMapPtr;
        using ConstXMapPtr              = typename ComposedOperator<SC,LO,GO,NO>::ConstXMapPtr;

        using XMultiVector              = typename ComposedOperator<SC,LO,GO,NO>::XMultiVector;
        using XMultiVectorPtr           = typename ComposedOperator<SC,LO,GO,NO>::XMultiVectorPtr;

        using SchwarzOperatorPtr        = typename ComposedOperator<SC,LO,GO,NO>::SchwarzOperatorPtr;
        using SchwarzOperatorPtrVec     = typename ComposedOperator<SC,LO,GO,NO>::SchwarzOperatorPtrVec;
        using SchwarzOperatorPtrVecPtr  = typename ComposedOperator<SC,LO,GO,NO>::SchwarzOperatorPtrVecPtr;

        using UN                        = typename ComposedOperator<SC,LO,GO,NO>::UN;

        using BoolVec                   = typename ComposedOperator<SC,LO,GO,NO>::BoolVec;

        using ParameterListPtr          = typename ComposedOperator<SC,LO,GO,NO>::ParameterListPtr;

    public:
        //Import the constructors from the base class
        using ComposedOperator<SC,LO,GO,NO>::ComposedOperator;


        void preApplyCoarse(XMultiVector &x,
                            XMultiVector &y);

        virtual void apply(const XMultiVector &x,
                           XMultiVector &y,
                           bool usePreconditionerOnly,
                           ETransp mode=NO_TRANS,
                           SC alpha=ScalarTraits<SC>::one(),
                           SC beta=ScalarTraits<SC>::zero()) const;

    protected:
        // Additional Temp Vector for apply()
        mutable XMultiVectorPtr YTmp_;
    };

}

#endif
