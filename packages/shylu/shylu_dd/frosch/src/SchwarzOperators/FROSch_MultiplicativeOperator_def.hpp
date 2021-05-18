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

#ifndef _FROSCH_MULTIPLICATIVEOPERATOR_DEF_HPP
#define _FROSCH_MULTIPLICATIVEOPERATOR_DEF_HPP

#include <FROSch_MultiplicativeOperator_decl.hpp>


namespace FROSch {

    using namespace std;
    using namespace Teuchos;
    using namespace Xpetra;

    //TODO: The constructors are still chaos, Why is there no constructor for sum with these parameters?
    // Any ideas how to make this more elegant? Line 61 and 70ff. are still double
    template <class SC,class LO,class GO,class NO>
    MultiplicativeOperator<SC,LO,GO,NO>::MultiplicativeOperator(ConstXMatrixPtr k,
                                                                ParameterListPtr parameterList) :
    CombinedOperator<SC,LO,GO,NO> (k, parameterList)
    {
        FROSCH_DETAILTIMER_START_LEVELID(multiplicativeOperatorTime, getOperatorName()+"::"+getOperatorName());
    }

    template <class SC,class LO,class GO,class NO>
    MultiplicativeOperator<SC,LO,GO,NO>::MultiplicativeOperator(ConstXMatrixPtr k,
                                                                SchwarzOperatorPtrVecPtr operators,
                                                                ParameterListPtr parameterList) :
    CombinedOperator<SC,LO,GO,NO> (k, parameterList)
    {
        FROSCH_DETAILTIMER_START_LEVELID(multiplicativeOperatorTime, getOperatorName()+"::"+getOperatorName());
        this->OperatorVector_.push_back(operators.at(0));
        for (unsigned i=1; i<operators.size(); i++) {
            FROSCH_ASSERT(operators[i]->OperatorDomainMap().SameAs(this->OperatorVector_[i]->OperatorDomainMap()),"The DomainMaps of the operators are not identical.");
            FROSCH_ASSERT(operators[i]->OperatorRangeMap().SameAs(this->OperatorVector_[i]->OperatorRangeMap()),"The RangeMaps of the operators are not identical.");

            this->OperatorVector_.push_back(operators[i]);
            this->EnableOperators_.push_back(true);
        }
    }

    template <class SC,class LO,class GO,class NO>
    MultiplicativeOperator<SC,LO,GO,NO>::~MultiplicativeOperator()
    {

    }

    template <class SC,class LO,class GO,class NO>
    void MultiplicativeOperator<SC,LO,GO,NO>::preApplyCoarse(XMultiVector &x,
                                                             XMultiVector &y)
    {
        FROSCH_DETAILTIMER_START_LEVELID(preApplyCoarseTime,getOperatorName()+"::preApplyCoarse");
        FROSCH_ASSERT(this->OperatorVector_.size()==2,"Should be a Two-Level Operator.");
        this->OperatorVector_[1]->apply(x,y,true);
    }

    // Y = alpha * A^mode * X + beta * Y
    template <class SC,class LO,class GO,class NO>
    void MultiplicativeOperator<SC,LO,GO,NO>::apply(const XMultiVector &x,
                                                    XMultiVector &y,
                                                    bool usePreconditionerOnly,
                                                    ETransp mode,
                                                    SC alpha,
                                                    SC beta) const
    {
        FROSCH_TIMER_START_LEVELID(applyTime,getOperatorName()+"::apply");
        FROSCH_ASSERT(usePreconditionerOnly,getOperatorName()+" can only be used as a preconditioner.");
        FROSCH_ASSERT(this->OperatorVector_.size()==2,"Should be a Two-Level Operator.");


        if (this->XTmp_.is_null()) this->XTmp_ = MultiVectorFactory<SC,LO,GO,NO>::Build(x.getMap(),x.getNumVectors());
        *(this->XTmp_) = x; // Need this for the case when x aliases y

        if (YTmp_.is_null()) XMultiVectorPtr YTmp_ = MultiVectorFactory<SC,LO,GO,NO>::Build(y.getMap(),y.getNumVectors());
        *YTmp_ = y; // for the second apply

        this->OperatorVector_[0]->apply(*(this->XTmp_),*YTmp_,true);

        this->K_->apply(*YTmp_,*(this->XTmp_));

        this->OperatorVector_[1]->apply(*(this->XTmp_),*(this->XTmp_),true);

        YTmp_->update(ScalarTraits<SC>::one(),*(this->XTmp_),-ScalarTraits<SC>::one());
        y.update(alpha,*YTmp_,beta);
    }

    template <class SC,class LO,class GO,class NO>
    string MultiplicativeOperator<SC,LO,GO,NO>::getOperatorName() const
    {
        return "MultiplicativeOperator";
    }
}

#endif
