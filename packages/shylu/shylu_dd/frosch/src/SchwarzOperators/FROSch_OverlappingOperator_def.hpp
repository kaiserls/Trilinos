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
            Combine_ = Averaging;
        } else if (!this->ParameterList_->get("Combine Values in Overlap","Restricted").compare("Full")) {
            Combine_ = Full;
        } else if (!this->ParameterList_->get("Combine Values in Overlap","Restricted").compare("Restricted")) {
            Combine_ = Restricted;
        }
        //TODO: Reworkt his with new ciombientype
        HarmonicOnOverlap_ = this->ParameterList_->get("HarmonicOnOverlap",false);
        FROSCH_ASSERT(HarmonicOnOverlap_ == (Combine_ == Full), "HarmonicOnOverlap can only be combined with CombineType=Full");
        //TODO: Only as preconditioner?!
        //TODO: Remove this after finishing harmonic overlap
        std::cout<<"Using my source code for harmonic overlap"<<HarmonicOnOverlap_<<std::endl;
    }

    template <class SC,class LO,class GO,class NO>
    OverlappingOperator<SC,LO,GO,NO>::~OverlappingOperator()
    {
        SubdomainSolver_.reset();
    }

    // Y = alpha * A^mode * X + beta * Y
    //TODO: Explain the different steps beforehand or in the code. Same could be achieved by creating
    // more subfunctions and giving them good names.
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
        if (XTmp_.is_null()) XTmp_ = MultiVectorFactory<SC,LO,GO,NO>::Build(x.getMap(),x.getNumVectors());
        *XTmp_ = x;

        //TODO: Implement harmonic overlap
        // HarmonicOnOverlap means, that the local solution should decay harmonic on the overlap: B(u,u)=0
        // Therefore we set the right hand side x (in the residual equation in a krylov space solver) to zero
        // for all entries in the overlap. This in done by examining the multiplicity of the node.
        std::cout<<"multiplicity"<<std::endl;
        if(HarmonicOnOverlap_){
            ConstSCVecPtr multi = Multiplicity_->getData(0);
            for (UN j=0; j<XTmp_->getNumVectors(); j++) {
                SCVecPtr values = XTmp_->getDataNonConst(j);
                for (UN i=0; i<values.size(); i++) {
                    if(multi[i]>1) values[i] = ScalarTraits<SC>::zero();
                    std::cout<<multi[i];
                }
                
            }
        }
        std::cout<<std::endl;
        //TODO: test with galeri, fem example
        // Create harmonic vector
        // const RCP<const XMultiVector> Bptr = rcp(&B,false);
        // XMultiVectorPtr Xharmonic =  MultiVectorFactory<SC,LO,GO,NO>::Build(XTmp_,Teuchos::DataAccess::Copy);
        // ConstSCVecPtr multi = Multiplicity_->getData(0);
        // for (UN j=0; j<Xharmonic->getNumVectors(); j++) {
        //     SCVecPtr values = Xharmonic->getDataNonConst(j);
        //     for (UN i=0; i<values.size(); i++) {
        //         if(multi[i]>1) values[i] = ScalarTraits<SC>::zero();
        //     }
        // }

        // fixpunkt etc.
        if (!usePreconditionerOnly && mode == NO_TRANS) {
            this->K_->apply(x,*XTmp_,mode,ScalarTraits<SC>::one(),ScalarTraits<SC>::zero());
        }
        // AH 11/28/2018: For Epetra, XOverlap_ will only have a view to the values of XOverlapTmp_. Therefore, xOverlapTmp should not be deleted before XOverlap_ is used.
        if (YOverlap_.is_null()) {//first time running appy will create this vector
            YOverlap_ = MultiVectorFactory<SC,LO,GO,NO>::Build(OverlappingMatrix_->getDomainMap(),x.getNumVectors());
        } else { //switch from global to local communicator -> sure that no communication happens
            YOverlap_->replaceMap(OverlappingMatrix_->getDomainMap());
        }

        //TODO: Explain a little bit more what is happening here
        // AH 11/28/2018: replaceMap does not update the GlobalNumRows. Therefore, we have to create a new MultiVector on the serial Communicator. In Epetra, we can prevent to copy the MultiVector.
        if (XTmp_->getMap()->lib() == UseEpetra) {
#ifdef HAVE_XPETRA_EPETRA
            if (XOverlapTmp_.is_null()) XOverlapTmp_ = MultiVectorFactory<SC,LO,GO,NO>::Build(OverlappingMap_,x.getNumVectors());
            XOverlapTmp_->doImport(*XTmp_,*Scatter_,INSERT);
            const RCP<const EpetraMultiVectorT<GO,NO> > xEpetraMultiVectorXOverlapTmp = rcp_dynamic_cast<const EpetraMultiVectorT<GO,NO> >(XOverlapTmp_);
            RCP<Epetra_MultiVector> epetraMultiVectorXOverlapTmp = xEpetraMultiVectorXOverlapTmp->getEpetra_MultiVector();
            const RCP<const EpetraMapT<GO,NO> >& xEpetraMap = rcp_dynamic_cast<const EpetraMapT<GO,NO> >(OverlappingMatrix_->getRangeMap());
            Epetra_BlockMap epetraMap = xEpetraMap->getEpetra_BlockMap();
            double *A;
            int MyLDA;
            epetraMultiVectorXOverlapTmp->ExtractView(&A,&MyLDA);
            RCP<Epetra_MultiVector> epetraMultiVectorXOverlap(new Epetra_MultiVector(::View,epetraMap,A,MyLDA,x.getNumVectors()));
            XOverlap_ = RCP<EpetraMultiVectorT<GO,NO> >(new EpetraMultiVectorT<GO,NO>(epetraMultiVectorXOverlap));
#else
            FROSCH_ASSERT(false,"HAVE_XPETRA_EPETRA not defined.");
#endif
        } else {
            if (XOverlap_.is_null()) {
                XOverlap_ = MultiVectorFactory<SC,LO,GO,NO>::Build(OverlappingMap_,x.getNumVectors());
            } else {
                XOverlap_->replaceMap(OverlappingMap_);//global map
            }
            XOverlap_->doImport(*XTmp_,*Scatter_,INSERT);
            XOverlap_->replaceMap(OverlappingMatrix_->getRangeMap());// local map
        }
        SubdomainSolver_->apply(*XOverlap_,*YOverlap_,mode,ScalarTraits<SC>::one(),ScalarTraits<SC>::zero());
        YOverlap_->replaceMap(OverlappingMap_);

        XTmp_->putScalar(ScalarTraits<SC>::zero());
        ConstXMapPtr yMap = y.getMap();
        ConstXMapPtr yOverlapMap = YOverlap_->getMap();
        //TODO: harmonic sollte nicht mit restricted oder averaging vervendet werden!
        // apply großteils bei mir anders -> evtl eigene klasse abletien, oder restrictionPorolongation in subfunction
        if (Combine_ == Restricted) {
#if defined(HAVE_XPETRA_KOKKOS_REFACTOR) && defined(HAVE_XPETRA_TPETRA)
            if (XTmp_->getMap()->lib() == UseTpetra) {
                auto yLocalMap = yMap->getLocalMap();
                auto yLocalOverlapMap = yOverlapMap->getLocalMap();
                // run local restriction on execution space defined by local-map
                using XMap            = typename SchwarzOperator<SC,LO,GO,NO>::XMap;
                using execution_space = typename XMap::local_map_type::execution_space;
                Kokkos::RangePolicy<execution_space> policy (0, yMap->getNodeNumElements());
                for (UN i=0; i<y.getNumVectors(); i++) {
                    auto yOverlapData_i = YOverlap_->getData(i);
                    auto xLocalData_i = XTmp_->getDataNonConst(i);
                    Kokkos::parallel_for(
                      "FROSch_OverlappingOperator::applyLocalRestriction", policy,
                      KOKKOS_LAMBDA(const int j) {
                        GO gID = yLocalMap.getGlobalElement(j);
                        LO lID = yLocalOverlapMap.getLocalElement(gID);
                        xLocalData_i[j] = yOverlapData_i[lID];
                      });
                }
                Kokkos::fence();
            } else
#endif
            {
                GO globID = 0;
                LO localID = 0;
                for (UN i=0; i<y.getNumVectors(); i++) {
                    ConstSCVecPtr yOverlapData_i = YOverlap_->getData(i);
                    for (UN j=0; j<yMap->getNodeNumElements(); j++) {
                        globID = yMap->getGlobalElement(j);
                        localID = yOverlapMap->getLocalElement(globID);
                        XTmp_->getDataNonConst(i)[j] = yOverlapData_i[localID];
                    }
                }
            }
        } else {
            XTmp_->doExport(*YOverlap_,*Scatter_,ADD);
        }
        if (Combine_ == Averaging) {
            ConstSCVecPtr scaling = Multiplicity_->getData(0);
            for (UN j=0; j<XTmp_->getNumVectors(); j++) {
                SCVecPtr values = XTmp_->getDataNonConst(j);
                for (UN i=0; i<values.size(); i++) {
                    values[i] = values[i] / scaling[i];
                }
            }
        }

        if (!usePreconditionerOnly && mode != NO_TRANS) {
            this->K_->apply(*XTmp_,*XTmp_,mode,ScalarTraits<SC>::one(),ScalarTraits<SC>::zero());
        }
        y.update(alpha,*XTmp_,beta);
    }

    template <class SC,class LO,class GO,class NO>
    int OverlappingOperator<SC,LO,GO,NO>::initializeOverlappingOperator()
    {
        FROSCH_DETAILTIMER_START_LEVELID(initializeOverlappingOperatorTime,"OverlappingOperator::initializeOverlappingOperator");
        Scatter_ = ImportFactory<LO,GO,NO>::Build(this->getDomainMap(),OverlappingMap_);
        // Calculate multiplicity if needed
        if (Combine_ == Averaging || HarmonicOnOverlap_) {
            Multiplicity_ = MultiVectorFactory<SC,LO,GO,NO>::Build(this->getRangeMap(),1);
            XMultiVectorPtr multiplicityRepeated;
            multiplicityRepeated = MultiVectorFactory<SC,LO,GO,NO>::Build(OverlappingMap_,1);
            multiplicityRepeated->putScalar(ScalarTraits<SC>::one());
            XExportPtr multiplicityExporter = ExportFactory<LO,GO,NO>::Build(multiplicityRepeated->getMap(),this->getRangeMap());
            Multiplicity_->doExport(*multiplicityRepeated,*multiplicityExporter,ADD);
        }

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
}

#endif
