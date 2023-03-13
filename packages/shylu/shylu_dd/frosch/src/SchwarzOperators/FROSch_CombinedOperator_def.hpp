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

#ifndef _FROSCH_COMBINEDOPERATOR_DEF_HPP
#define _FROSCH_COMBINEDOPERATOR_DEF_HPP

#include <FROSch_ComposedOperator_decl.hpp>


namespace FROSch {

    using namespace std;
    using namespace Teuchos;
    using namespace Xpetra;

    template <class SC,class LO,class GO,class NO>
    ComposedOperator<SC,LO,GO,NO>::ComposedOperator(CommPtr comm) :
    SchwarzOperator<SC,LO,GO,NO> (comm)
    {
        FROSCH_DETAILTIMER_START_LEVELID(composedOperatorTime, "ComposedOperator::ComposedOperator");
    }

    template <class SC,class LO,class GO,class NO>
    ComposedOperator<SC,LO,GO,NO>::ComposedOperator(SchwarzOperatorPtrVecPtr operators) :
    SchwarzOperator<SC,LO,GO,NO> (operators[0]->getRangeMap()->getComm())
    {
        FROSCH_DETAILTIMER_START_LEVELID(composedOperatorTime, "ComposedOperator::ComposedOperator");
        FROSCH_ASSERT(operators.size()>0,"operators.size()<=0");
        OperatorVector_.push_back(operators[0]);
        //TODO: Why isn't true pushed to enable operators?
        for (unsigned i=1; i<operators.size(); i++) {
            FROSCH_ASSERT(operators[i]->OperatorDomainMap().SameAs(OperatorVector_[0]->OperatorDomainMap()),"The DomainMaps of the operators are not identical.");
            FROSCH_ASSERT(operators[i]->OperatorRangeMap().SameAs(OperatorVector_[0]->OperatorRangeMap()),"The RangeMaps of the operators are not identical.");

            OperatorVector_.push_back(operators[i]);
            EnableOperators_.push_back(true);
        }
    }

    template <class SC,class LO,class GO,class NO>
    ComposedOperator<SC,LO,GO,NO>::~ComposedOperator()
    {

    }

    template <class SC,class LO,class GO,class NO>
    int ComposedOperator<SC,LO,GO,NO>::initialize()
    {
        if (this->Verbose_) {
            FROSCH_ASSERT(false,"ERROR: Each of the Operators has to be initialized manually.");
        }
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int ComposedOperator<SC,LO,GO,NO>::initialize(ConstXMapPtr repeatedMap)
    {
        if (this->Verbose_) {
            FROSCH_ASSERT(false,"ERROR: Each of the Operators has to be initialized manually.");
        }
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int ComposedOperator<SC,LO,GO,NO>::compute()
    {
        if (this->Verbose_) {
            FROSCH_ASSERT(false,"ERROR: Each of the Operators has to be computed manually.");
        }
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    typename ComposedOperator<SC,LO,GO,NO>::ConstXMapPtr ComposedOperator<SC,LO,GO,NO>::getDomainMap() const
    {
        return OperatorVector_[0]->getDomainMap();
    }

    template <class SC,class LO,class GO,class NO>
    typename ComposedOperator<SC,LO,GO,NO>::ConstXMapPtr ComposedOperator<SC,LO,GO,NO>::getRangeMap() const
    {
        return OperatorVector_[0]->getRangeMap();
    }

    template <class SC,class LO,class GO,class NO>
    void ComposedOperator<SC,LO,GO,NO>::describe(FancyOStream &out,
                                            const EVerbosityLevel verbLevel) const
    {
        FROSCH_ASSERT(false,"describe() has to be implemented properly...");
    }

    template <class SC,class LO,class GO,class NO>
    string ComposedOperator<SC,LO,GO,NO>::description() const
    {
        string labelString = "ComposedOperator: ";

        for (UN i=0; i<OperatorVector_.size(); i++) {
            labelString += OperatorVector_[i]->description();
            if (i<OperatorVector_.size()-1) {
                labelString += ",";
            }
        }
        return labelString;
    }

    //! Add a SchwarzOperator to the combination
    template <class SC,class LO,class GO,class NO>
    int ComposedOperator<SC,LO,GO,NO>::addOperator(SchwarzOperatorPtr op)
    {
        FROSCH_DETAILTIMER_START_LEVELID(addOperatorTime, "ComposedOperator::addOperator");
        // Check if the new operator is compatible with the existing one(s).
        int ret = 0;
        if (OperatorVector_.size()>0) {
            if (!op->getDomainMap()->isSameAs(*OperatorVector_[0]->getDomainMap())) {
                if (this->Verbose_) cerr <<  "ComposedOperator<SC,LO,GO,NO>::addOperator(SchwarzOperatorPtr op)\t\t!op->getDomainMap().isSameAs(OperatorVector_[0]->getDomainMap())\n";
                ret -= 1;
            }
            if (!op->getRangeMap()->isSameAs(*OperatorVector_[0]->getRangeMap())) {
                if (this->Verbose_) cerr <<  "ComposedOperator<SC,LO,GO,NO>::addOperator(SchwarzOperatorPtr op)\t\t!op->getRangeMap().isSameAs(OperatorVector_[0]->getRangeMap())\n";
                ret -= 10;
            }
        }
        // Add the operator, compatibility is encoded in return type (!=0 for invalid)
        OperatorVector_.push_back(op);
        EnableOperators_.push_back(true);
        return ret;
    }

    //! Add a vector/list of SchwarzOperators to the combination.
    template <class SC,class LO,class GO,class NO>
    int ComposedOperator<SC,LO,GO,NO>::addOperators(SchwarzOperatorPtrVecPtr operators)
    {
        FROSCH_DETAILTIMER_START_LEVELID(addOperatorsTime, "ComposedOperator::addOperators");
        int ret = 0;
        for (UN i=1; i<operators.size(); i++) {
            if (0>addOperator(operators[i])) ret -= pow(10,i);//Injective encoding of error value to added operator
        }
        return ret;
    }

    //! Replace a SchwarzOperator with specific id.
    template <class SC,class LO,class GO,class NO>
    int ComposedOperator<SC,LO,GO,NO>::resetOperator(UN iD,
                                                SchwarzOperatorPtr op)
    {
        FROSCH_DETAILTIMER_START_LEVELID(resetOperatorTime, "ComposedOperator::resetOperator");
        FROSCH_ASSERT(iD<OperatorVector_.size(),"iD exceeds the length of the OperatorVector_");
        int ret = 0;
        if (!op->getDomainMap().isSameAs(OperatorVector_[0]->getDomainMap())) {
            if (this->Verbose_) cerr <<  "ComposedOperator<SC,LO,GO,NO>::addOperator(SchwarzOperatorPtr op)\t\t!op->getDomainMap().isSameAs(OperatorVector_[0]->getDomainMap())\n";
            ret -= 1;
        }
        if (!op->getRangeMap().isSameAs(OperatorVector_[0]->getRangeMap())) {
            if (this->Verbose_) cerr <<  "ComposedOperator<SC,LO,GO,NO>::addOperator(SchwarzOperatorPtr op)\t\t!op->getRangeMap().isSameAs(OperatorVector_[0]->getRangeMap())\n";
            ret -= 10;
        }
        OperatorVector_[iD] = op;
        return ret;
    }

    //! Set the status of a SchwarzOperator with specific id.
    //! Disabled operators will be skipped in apply.
    template <class SC,class LO,class GO,class NO>
    int ComposedOperator<SC,LO,GO,NO>::enableOperator(UN iD,
                                                 bool enable)
    {
        FROSCH_DETAILTIMER_START_LEVELID(enableOperatorTime, "ComposedOperator::enableOperator");
        EnableOperators_[iD] = enable;
        return 0;
    }

    //! Returns the number of individual SchwarzOperators composed in this operator
    template <class SC,class LO,class GO,class NO>
    typename ComposedOperator<SC,LO,GO,NO>::UN ComposedOperator<SC,LO,GO,NO>::getNumOperators()
    {
        return OperatorVector_.size();
    }

    //! Preapply coarse allows to apply the coarse operator before the first level.
    //TODO: Results in an symmetric operator for a one-level preconditioner???
    //! Why is this not handled by adding the OperatorPtr to the vector againg?
    template <class SC,class LO,class GO,class NO>
    void ComposedOperator<SC,LO,GO,NO>::preApplyCoarse(XMultiVector &x,
                                XMultiVector &y) 
    {
        FROSCH_ASSERT(false,"preApplyCoarse(XMultiVectorPtr &x) only implemented for MultiplicativeOperator.")
    }
    
    //! Reset matrix for the composed operator and all operators contained in the OperatorVector
    template <class SC,class LO,class GO,class NO>
    void ComposedOperator<SC,LO,GO,NO>::resetMatrix(ConstXMatrixPtr &k)
    {
        SchwarzOperator<SC,LO,GO,NO>::resetMatrix(k);
        for (UN i=1; i<OperatorVector_.size(); i++) {
            OperatorVector_[i]->resetMatrix(k);
        }
    }
}

#endif
