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

#ifndef _FROSCH_Mapper_DEF_HPP
#define _FROSCH_Mapper_DEF_HPP

#include <FROSch_Mapper_decl.hpp>


namespace FROSch {

    using namespace std;
    using namespace Teuchos;
    using namespace Xpetra;

    template <class SC,class LO,class GO,class NO>
    Mapper<SC,LO,GO,NO>::Mapper(ConstXMapPtr uniqueMap, ConstXMapPtr overlappingMap,XMultiVectorPtr  multiplicity)
    : UniqueMap_(uniqueMap), OverlappingMap_(overlappingMap), Multiplicity_(multiplicity) //inherit from describeable?
    {
        RCP<const Comm<LO> > serialComm = rcp(new MpiComm<LO>(MPI_COMM_SELF));
        OverlappingMapLocal_ = Xpetra::MapFactory<LO,GO,NO>::Build(OverlappingMap_->lib(),OverlappingMap_->getNodeNumElements(),0,serialComm);
        Import_ = ImportFactory<LO,GO,NO>::Build(UniqueMap_,OverlappingMap_);
    }

    template <class SC,class LO,class GO,class NO>
    Mapper<SC,LO,GO,NO>::~Mapper(){}


    template <class SC,class LO,class GO,class NO>
    void Mapper<SC,LO,GO,NO>::setLocalMap(XMultiVector &globalVector){
        globalVector.replaceMap(OverlappingMapLocal_);
    }

    template <class SC,class LO,class GO,class NO>
    void Mapper<SC,LO,GO,NO>::setGlobalMap(XMultiVector &localVector){
        localVector.replaceMap(OverlappingMap_);
    }


    template <class SC,class LO,class GO,class NO>
    int Mapper<SC,LO,GO,NO>::restrict(const XMultiVectorPtr source, XMultiVectorPtr & target)
    {
        FROSCH_TIMER_START_LEVELID(applyTime,"Mapper::restrict");
        if (source->getMap()->lib() == UseEpetra) { // AH 11/28/2018: For Epetra, XOverlap_ will only have a view to the values of XOverlapTmp_. Therefore, xOverlapTmp should not be deleted before XOverlap_ is used.
#ifdef HAVE_XPETRA_EPETRA
            if (XOverlapTmp_.is_null()) XOverlapTmp_ = MultiVectorFactory<SC,LO,GO,NO>::Build(OverlappingMap_,source->getNumVectors());
            XOverlapTmp_->doImport(*source,*Import_,INSERT);
            const RCP<const EpetraMultiVectorT<GO,NO> > xEpetraMultiVectorXOverlapTmp = rcp_dynamic_cast<const EpetraMultiVectorT<GO,NO> >(XOverlapTmp_);
            RCP<Epetra_MultiVector> epetraMultiVectorXOverlapTmp = xEpetraMultiVectorXOverlapTmp->getEpetra_MultiVector();
            const RCP<const EpetraMapT<GO,NO> >& xEpetraMap = rcp_dynamic_cast<const EpetraMapT<GO,NO> >(OverlappingMapLocal_);
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
            target->doImport(*source,*Import_,INSERT);//unique source to overlapping target
            target->replaceMap(OverlappingMapLocal_);//global to local communicator, to prevent subdomain solvers communication
        }
        return 0;
    }


    template <class SC,class LO,class GO,class NO>
    int Mapper<SC,LO,GO,NO>::prolongate(const XMultiVectorPtr source, XMultiVectorPtr & target)
    {
        target->putScalar(ScalarTraits<SC>::zero());
        
        if (Combine_ == Restricted) {
            ConstXMapPtr overlapMap = source->getMap();
#if defined(HAVE_XPETRA_KOKKOS_REFACTOR) && defined(HAVE_XPETRA_TPETRA)
            if (target->getMap()->lib() == UseTpetra) {
                auto localMap = UniqueMap_->getLocalMap();
                auto localOverlapMap = overlapMap->getLocalMap();
                // run local restriction on execution space defined by local-map
                using execution_space = typename XMap::local_map_type::execution_space;
                Kokkos::RangePolicy<execution_space> policy (0, UniqueMap_->getNodeNumElements());
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
                    for (UN j=0; j<UniqueMap_->getNodeNumElements(); j++) {
                        globID = UniqueMap_->getGlobalElement(j);
                        localID = overlapMap->getLocalElement(globID);
                        target->getDataNonConst(i)[j] = overlapData_i[localID];
                    }
                }
            }
        } else { // All modes, excluding restricted
            target->doExport(*source,*Import_,ADD);
        }
        
        // Divide the result by the number of subdomains which contributed to this value
        if (Combine_ == Averaging) {
            ConstSCVecPtr scaling = Multiplicity_->getData(0);
            for (UN j=0; j<target->getNumVectors(); j++) {
                SCVecPtr values = target->getDataNonConst(j);
                for (UN i=0; i<values.size(); i++) {
                    values[i] = values[i] / scaling[i];
                }
            }
        }
        return 0;
    }
}

#endif