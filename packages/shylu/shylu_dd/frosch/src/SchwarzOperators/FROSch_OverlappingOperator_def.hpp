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
            Combine_ = CombinationType::Averaging;
        } else if (!this->ParameterList_->get("Combine Values in Overlap","Restricted").compare("Full")) {
            Combine_ = CombinationType::Full;
        } else if (!this->ParameterList_->get("Combine Values in Overlap","Restricted").compare("Restricted")) {
            Combine_ = CombinationType::Restricted;
        }
    }

    template <class SC,class LO,class GO,class NO>
    OverlappingOperator<SC,LO,GO,NO>::~OverlappingOperator()
    {
        SubdomainSolver_.reset();
    }

    //TODO: For harmonic(but not for the presolve/afterSolve): use altered scatter importer to reduce communication to nonoverlapping, restrictedMode?
    template <class SC,class LO,class GO,class NO>
    void OverlappingOperator<SC,LO,GO,NO>::restrictFromInto(const XMultiVectorPtr source, XMultiVectorPtr & target) const
    {
        FROSCH_TIMER_START_LEVELID(applyTime,"OverlappingOperator::restrict");
        FROSCH_ASSERT(this->IsInitialized_,"FROSch::OverlappingOperator: OverlappingOperator has to be initialized before calling apply()");

        if (source->getMap()->lib() == UseEpetra) { // AH 11/28/2018: For Epetra, XOverlap_ will only have a view to the values of XOverlapTmp_. Therefore, xOverlapTmp should not be deleted before XOverlap_ is used.
#ifdef HAVE_XPETRA_EPETRA
            if (XOverlapTmp_.is_null()) XOverlapTmp_ = MultiVectorFactory<SC,LO,GO,NO>::Build(OverlappingMap_,source->getNumVectors());
            XOverlapTmp_->doImport(*source,*Scatter_,INSERT);
            const RCP<const EpetraMultiVectorT<GO,NO> > xEpetraMultiVectorXOverlapTmp = rcp_dynamic_cast<const EpetraMultiVectorT<GO,NO> >(XOverlapTmp_);
            RCP<Epetra_MultiVector> epetraMultiVectorXOverlapTmp = xEpetraMultiVectorXOverlapTmp->getEpetra_MultiVector();
            const RCP<const EpetraMapT<GO,NO> >& xEpetraMap = rcp_dynamic_cast<const EpetraMapT<GO,NO> >(OverlappingMatrix_->getRangeMap());
            Epetra_BlockMap epetraMap = xEpetraMap->getEpetra_BlockMap();
            double *A;
            int MyLDA;
            epetraMultiVectorXOverlapTmp->ExtractView(&A,&MyLDA);
            RCP<Epetra_MultiVector> epetraMultiVectorXOverlap(new Epetra_MultiVector(::View,epetraMap,A,MyLDA,source->getNumVectors()));
            target = RCP<EpetraMultiVectorT<GO,NO> >(new EpetraMultiVectorT<GO,NO>(epetraMultiVectorXOverlap));
#else
            FROSCH_ASSERT(false,"HAVE_XPETRA_EPETRA not defined.");
#endif
        } else {
            // Do Import into overlapping local vector
            if (target.is_null()) {
                target = MultiVectorFactory<SC,LO,GO,NO>::Build(OverlappingMap_,source->getNumVectors());
            } else {
                target->replaceMap(OverlappingMap_);// to global communicator
            }
            target->doImport(*source,*Scatter_,INSERT);//unique source to overlapping target
            target->replaceMap(OverlappingMatrix_->getRangeMap());//global to local communicator, to prevent subdomain solvers communication
        }
    }

    //TODO: for harmonic (but not pre/aftersolve)
    //For restricted: Replace xmultivector by map containing the nodes which should be imported.
    //                This would be getNonoverlappingNodesMap orSomeslightlymodified version for the restricted paper.
    //For all other: Use altered scatter importer to reduce communication to nonoverlapping
    //For averaging (additional): disable it, multiplicity will be never bigger than 1 for the nodes where the import happens
    template <class SC,class LO,class GO,class NO>
    void OverlappingOperator<SC,LO,GO,NO>::prolongateFromInto(const XMultiVectorPtr source, XMultiVectorPtr target, const ConstXMapPtr uniqueMap) const
    {
        FROSCH_TIMER_START_LEVELID(applyTime,"OverlappingOperator::prolongate");
        FROSCH_ASSERT(this->IsInitialized_,"FROSch::OverlappingOperator: OverlappingOperator has to be initialized before calling prolongate()");
        target->putScalar(ScalarTraits<SC>::zero());
        
        if (Combine_ == CombinationType::Restricted) {
            ConstXMapPtr overlapMap = source->getMap();
#if defined(HAVE_XPETRA_KOKKOS_REFACTOR) && defined(HAVE_XPETRA_TPETRA)
            if (target->getMap()->lib() == UseTpetra) {
                auto localMap = uniqueMap->getLocalMap();
                auto localOverlapMap = overlapMap->getLocalMap();
                // run local restriction on execution space defined by local-map
                using XMap            = typename SchwarzOperator<SC,LO,GO,NO>::XMap;
                using execution_space = typename XMap::local_map_type::execution_space;
                Kokkos::RangePolicy<execution_space> policy (0, uniqueMap->getNodeNumElements());
                for (UN i=0; i<target->getNumVectors(); i++) {
                    auto sourceData_i = source->getData(i);
                    auto targetLocalData_i = target->getDataNonConst(i);
                    Kokkos::parallel_for(
                      "FROSch_OverlappingOperator::applyLocalRestriction", policy,
                      KOKKOS_LAMBDA(const int j) {
                        GO gID = localMap.getGlobalElement(j);
                        LO lID = localOverlapMap.getLocalElement(gID);
                        targetLocalData_i[j] = sourceData_i[lID];
                      });
                }
                Kokkos::fence();
            } else
#endif
            {
                GO globID = 0;
                LO localID = 0;
                for (UN i=0; i<target->getNumVectors(); i++) {
                    ConstSCVecPtr overlapData_i = source->getData(i);
                    for (UN j=0; j<uniqueMap->getNodeNumElements(); j++) {
                        globID = uniqueMap->getGlobalElement(j);
                        localID = overlapMap->getLocalElement(globID);
                        target->getDataNonConst(i)[j] = overlapData_i[localID];
                    }
                }
            }
        } else { // All modes, excluding restricted
            target->doExport(*source,*Scatter_,ADD);
        }
        
        // Divide the result by the number of subdomains which contributed to this value
        if (Combine_ == CombinationType::Averaging) {
            ConstSCVecPtr scaling = Multiplicity_->getData(0);
            for (UN j=0; j<target->getNumVectors(); j++) {
                SCVecPtr values = target->getDataNonConst(j);
                for (UN i=0; i<values.size(); i++) {
                    values[i] = values[i] / scaling[i];
                }
            }
        }
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
        if (Combine_ == CombinationType::Averaging) {
            calculateMultiplicity();
        }
        Mapper_= rcp(new Mapper<SC,LO,GO,NO>(this->getDomainMap(),OverlappingMap_, Multiplicity_, Combine_));
        return 0; // RETURN VALUE
    }

    //! Essentially prepares the subdomain solver to be used
    template <class SC,class LO,class GO,class NO>
    int OverlappingOperator<SC,LO,GO,NO>::computeOverlappingOperator()
    {
        FROSCH_DETAILTIMER_START_LEVELID(computeOverlappingOperatorTime,"OverlappingOperator::computeOverlappingOperator");

        updateLocalOverlappingMatrices();

        bool reuseSymbolicFactorization = this->ParameterList_->get("Reuse: Symbolic Factorization",true);
        if (!this->IsComputed_) {
            reuseSymbolicFactorization = false;
        }

        if (!reuseSymbolicFactorization) {
            if (this->IsComputed_ && this->Verbose_) cout << "FROSch::OverlappingOperator : Recomputing the Symbolic Factorization" << endl;
            SubdomainSolver_ = SolverFactory<SC,LO,GO,NO>::Build(OverlappingMatrix_,
                                                                 sublist(this->ParameterList_,"Solver"),
                                                                 string("Solver (Level ") + to_string(this->LevelID_) + string(")"));
            SubdomainSolver_->initialize();
        } else {
            FROSCH_ASSERT(!SubdomainSolver_.is_null(),"FROSch::OverlappingOperator: SubdomainSolver_.is_null()");
            SubdomainSolver_->updateMatrix(OverlappingMatrix_,true);
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
