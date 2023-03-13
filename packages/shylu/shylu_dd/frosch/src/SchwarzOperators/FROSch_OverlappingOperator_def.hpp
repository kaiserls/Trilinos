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

#ifndef _FROSCH_OVERLAPPINGOPERATOR_DEF_HPP
#define _FROSCH_OVERLAPPINGOPERATOR_DEF_HPP

#include <FROSch_OverlappingOperator_decl.hpp>
#include <FROSch_Debugging.hpp>

namespace FROSch {

    using namespace std;
    using namespace Teuchos;
    using namespace Xpetra;

    template <class SC,class LO,class GO,class NO>
    OverlappingOperator<SC,LO,GO,NO>::OverlappingOperator(ConstXMatrixPtr k,
                                                          ParameterListPtr parameterList) :
    SchwarzOperator<SC,LO,GO,NO> (k,parameterList)
    {
        FROSCH_DETAILTIMER_START_LEVELID(overlappingOperatorTime,"OverlappingOperator::OverlappingOperator");
        if (!this->ParameterList_->get("Combine Values in Overlap","Restricted").compare("Averaging")) {
            Combine_ = OverlapCombinationType::Averaging;
        } else if (!this->ParameterList_->get("Combine Values in Overlap","Restricted").compare("Full")) {
            Combine_ = OverlapCombinationType::Full;
        } else if (!this->ParameterList_->get("Combine Values in Overlap","Restricted").compare("Restricted")) {
            Combine_ = OverlapCombinationType::Restricted;
        }
    }

    template <class SC,class LO,class GO,class NO>
    OverlappingOperator<SC,LO,GO,NO>::~OverlappingOperator()
    {
        SubdomainSolver_.reset();
    }

    template <class SC,class LO,class GO,class NO>
    void OverlappingOperator<SC,LO,GO,NO>::restrictFromInto(const XMultiVectorPtr source, XMultiVectorPtr & target) const
    {
        FROSCH_TIMER_START_LEVELID(applyTime,"OverlappingOperator::restrict");
        FROSCH_ASSERT(this->IsInitialized_,"FROSch::OverlappingOperator: OverlappingOperator has to be initialized before calling apply()");
        Mapper_->restrict(source, target);
    }

    //For restricted: Replace xmultivector by map containing the nodes which should be imported.
    //                This would be getNonoverlappingNodesMap orSomeslightlymodified version for the restricted paper.
    //For all other: Use altered scatter importer to reduce communication to nonoverlapping
    //For averaging (additional): disable it, multiplicity will be never bigger than 1 for the nodes where the import happens
    template <class SC,class LO,class GO,class NO>
    void OverlappingOperator<SC,LO,GO,NO>::prolongateFromInto(const XMultiVectorPtr source, XMultiVectorPtr target, const ConstXMapPtr uniqueMap) const
    {
        FROSCH_TIMER_START_LEVELID(applyTime,"OverlappingOperator::prolongate");
        FROSCH_ASSERT(this->IsInitialized_,"FROSch::OverlappingOperator: OverlappingOperator has to be initialized before calling prolongate()");
        Mapper_->prolongate(source, target);
    }

    //! Y = alpha * A^mode * X + beta * Y
    template <class SC,class LO,class GO,class NO>
    void OverlappingOperator<SC,LO,GO,NO>::apply(const XMultiVector &x,
                                                 XMultiVector &y,
                                                 bool usePreconditionerOnly,
                                                 ETransp mode,
                                                 SC alpha,
                                                 SC beta) const
    {
        FROSCH_TIMER_START_LEVELID(applyTime,"OverlappingOperator::apply");
        FROSCH_ASSERT(this->IsComputed_,"FROSch::OverlappingOperator: OverlappingOperator has to be computed before calling apply()");
        // Temporary vectors for input/result of local subdomainSolver and matrix vector operations
        if (YOverlap_.is_null()) {
            YOverlap_ = MultiVectorFactory<SC,LO,GO,NO>::Build(OverlappingMatrix_->getDomainMap(),x.getNumVectors());
        } else {
            YOverlap_->replaceMap(OverlappingMatrix_->getDomainMap());
        }
        if (XTmp_.is_null()) XTmp_ = MultiVectorFactory<SC,LO,GO,NO>::Build(x.getMap(),x.getNumVectors());

        *XTmp_ = x;
        // Apply K first if the framework is not only used as preconditioner: P = M^-1 K
        if (!usePreconditionerOnly && mode == NO_TRANS) { // If mode != NO_TRANS it is applied at the end
            this->K_->apply(x,*XTmp_,mode,ScalarTraits<SC>::one(),ScalarTraits<SC>::zero());
        }
        
        restrictFromInto(XTmp_, XOverlap_); //Restrict into the local overlapping subdomain vector
        SubdomainSolver_->apply(*XOverlap_,*YOverlap_,mode,ScalarTraits<SC>::one(),ScalarTraits<SC>::zero());
        YOverlap_->replaceMap(OverlappingMap_);
        prolongateFromInto(YOverlap_, XTmp_,y.getMap());//y is there to provide a unique map for the restricted import

        if (!usePreconditionerOnly && mode != NO_TRANS) {
            this->K_->apply(*XTmp_,*XTmp_,mode,ScalarTraits<SC>::one(),ScalarTraits<SC>::zero());
        }
        y.update(alpha,*XTmp_,beta);
    }

    template <class SC,class LO,class GO,class NO>
    int OverlappingOperator<SC,LO,GO,NO>::calculateMultiplicity(){
        Multiplicity_ = MultiVectorFactory<SC,LO,GO,NO>::Build(this->getRangeMap(),1);
        XMultiVectorPtr multiplicityRepeated;
        multiplicityRepeated = MultiVectorFactory<SC,LO,GO,NO>::Build(OverlappingMap_,1);
        multiplicityRepeated->putScalar(ScalarTraits<SC>::one()); // Every node has at least multiplicity one
        XExportPtr multiplicityExporter = ExportFactory<LO,GO,NO>::Build(multiplicityRepeated->getMap(),this->getRangeMap());
        Multiplicity_->doExport(*multiplicityRepeated,*multiplicityExporter,ADD);
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int OverlappingOperator<SC,LO,GO,NO>::initializeOverlappingOperator()
    {
        FROSCH_DETAILTIMER_START_LEVELID(initializeOverlappingOperatorTime,"OverlappingOperator::initializeOverlappingOperator");
        Scatter_ = ImportFactory<LO,GO,NO>::Build(this->getDomainMap(),OverlappingMap_);
        // Calculate multiplicity if needed
        if (Combine_ == OverlapCombinationType::Averaging) {
            calculateMultiplicity();
        }
        Mapper_= rcp(new Mapper<SC,LO,GO,NO>(this->getDomainMap(),OverlappingMap_, Multiplicity_, Combine_));
        return 0; // RETURN VALUE
    }

    template <class SC,class LO,class GO,class NO>
    int OverlappingOperator<SC,LO,GO,NO>::initializeSubdomainSolver(ConstXMatrixPtr localMat)
    {
        FROSCH_DETAILTIMER_START_LEVELID(initializeSubdomainSolverTime,"OverlappingOperator::initializeSubdomainSolver");
        SubdomainSolver_ = SolverFactory<SC,LO,GO,NO>::Build(localMat,
                                                             sublist(this->ParameterList_,"Solver"),
                                                             string("Solver (Level ") + to_string(this->LevelID_) + string(")"));
        SubdomainSolver_->initialize();
        return 0; // RETURN VALUE
    }

    //! Essentially prepares the subdomain solver to be used
    template <class SC,class LO,class GO,class NO>
    int OverlappingOperator<SC,LO,GO,NO>::computeOverlappingOperator()
    {
        FROSCH_DETAILTIMER_START_LEVELID(computeOverlappingOperatorTime,"OverlappingOperator::computeOverlappingOperator");

        updateLocalOverlappingMatrices();
        bool reuseSymbolicFactorization = this->ParameterList_->get("Reuse: Symbolic Factorization",true);
        if (!reuseSymbolicFactorization || SubdomainSolver_.is_null()) {
            // initializeSubdomainSolver is called during symbolic only if reuseSymbolicFactorization=true
            // so if reuseSymbolicFactorization=false, we always call initializeSubdomainSolver 
            if (this->IsComputed_ && this->Verbose_) cout << "FROSch::OverlappingOperator : Recomputing the Symbolic Factorization" << endl;
            initializeSubdomainSolver(this->OverlappingMatrix_);
        } else if (this->IsComputed_) {
            // if !IsComputed, then this is the first timing calling "compute" after initializeSubdomainSolver is called in symbolic phase
            // so no need to do anything
            SubdomainSolver_->updateMatrix(this->OverlappingMatrix_,true);
        }
        this->IsComputed_ = true;
        return SubdomainSolver_->compute();
    }

    template <class SC,class LO,class GO,class NO>
    int OverlappingOperator<SC,LO,GO,NO>::updateLocalOverlappingMatrices()
    {
        FROSCH_DETAILTIMER_START_LEVELID(updateLocalOverlappingMatricesTime,"AlgebraicOverlappingOperator::updateLocalOverlappingMatrices");
        if (this->IsComputed_) { // already computed once and we want to recycle the information. That is why we reset OverlappingMatrix_ to K_, because K_ has been reset at this point
            this->OverlappingMatrix_ = this->K_;
        }
        this->OverlappingMatrix_ = ExtractLocalSubdomainMatrix(this->OverlappingMatrix_,this->OverlappingMap_);
        return 0;
    }
}

#endif
