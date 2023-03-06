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

    //TODO: Think about how to make this work in both cases, maybe two classes? Simple and complex mapper?
    template <class SC,class LO,class GO,class NO>
    Mapper<SC,LO,GO,NO>::Mapper(ConstXMapPtr uniqueMap, ConstXMapPtr overlappingMap, XMultiVectorPtr  multiplicity, CombinationType combine)
    : UniqueMap_(uniqueMap), OverlappingMap_(overlappingMap), Multiplicity_(multiplicity), Combine_(combine) //inherit from describeable?
    {
        RCP<const Comm<LO> > serialComm = rcp(new MpiComm<LO>(MPI_COMM_SELF));
        OverlappingMapLocal_ = Xpetra::MapFactory<LO,GO,NO>::Build(OverlappingMap_->lib(),OverlappingMap_->getNodeNumElements(),0,serialComm);
        Import_ = ImportFactory<LO,GO,NO>::Build(UniqueMap_,OverlappingMap_);
    }

    template <class SC,class LO,class GO,class NO>
    Mapper<SC,LO,GO,NO>::Mapper(ConstXMapPtr uniqueMap, ConstXMapPtr overlappingMap, ConstXMapPtr importMap, ConstXMapPtr exportMap, XMultiVectorPtr multiplicity, CombinationType combine)
    : UniqueMap_(uniqueMap), OverlappingMap_(overlappingMap), ImportMap_(importMap), ExportMap_(exportMap), Multiplicity_(multiplicity), Combine_(combine)
    {
        RCP<const Comm<LO> > serialComm = rcp(new MpiComm<LO>(MPI_COMM_SELF));
        OverlappingMapLocal_ = Xpetra::MapFactory<LO,GO,NO>::Build(OverlappingMap_->lib(),OverlappingMap_->getNodeNumElements(),0,serialComm);
        Import_ = ImportFactory<LO,GO,NO>::Build(UniqueMap_, ImportMap_);
        Export_ = ExportFactory<LO,GO,NO>::Build(ExportMap_, UniqueMap_);
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
        FROSCH_DETAILTIMER_START(restrict,"Restrict");
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
                target->putScalar(ScalarTraits<SC>::zero());
                target->replaceMap(OverlappingMap_);// to global communicator
            }
            target->doImport(*source,*Import_,INSERT);//unique source to overlapping target
            target->replaceMap(OverlappingMapLocal_);//global to local communicator, to prevent subdomain solvers communication
        }
        return 0;
    }


    template <class SC,class LO,class GO,class NO> //y , x
    int Mapper<SC,LO,GO,NO>::prolongate(const XMultiVectorPtr source, XMultiVectorPtr & target)
    {
         FROSCH_DETAILTIMER_START(prolongate,"Prolongate");
        target->putScalar(ScalarTraits<SC>::zero());
        
        if (Combine_ == Restricted) {
            ConstXMapPtr overlapMap = source->getMap();
#if defined(HAVE_XPETRA_KOKKOS_REFACTOR) && defined(HAVE_XPETRA_TPETRA)
            if (target->getMap()->lib() == UseTpetra) {
                auto localMap = UniqueMap_->getLocalMap();
                auto localOverlapMap = overlapMap->getLocalMap();
                // run local restriction on execution space defined by local-map
                using execution_space = typename XMap::local_map_type::execution_space;
                Kokkos::RangePolicy<execution_space> policy (0, UniqueMap_->getLocalNumElements());

                using xTMVector    = Xpetra::TpetraMultiVector<SC,LO,GO,NO>;
                // Xpetra wrapper for Tpetra MV
                auto sourceXTpetraMVector = rcp_dynamic_cast<const xTMVector>(source, true);
                auto targetXTpetraMVector = rcp_dynamic_cast<      xTMVector>(target, true);
                // Tpetra MV
                auto sourceTpetraMVector = sourceXTpetraMVector->getTpetra_MultiVector();
                auto targetTpetraMVector = targetXTpetraMVector->getTpetra_MultiVector();
                // View
                auto sourceView = sourceTpetraMVector->getLocalViewDevice(Tpetra::Access::ReadOnly);
                auto targetView = targetTpetraMVector->getLocalViewDevice(Tpetra::Access::ReadWrite);
                
                for (UN j=0; j<target->getNumVectors(); j++) {
                    Kokkos::parallel_for(
                      "FROSch_OverlappingOperator::applyLocalRestriction", policy,
                      KOKKOS_LAMBDA(const int i) {
                        GO gID = localMap.getGlobalElement(i);
                        LO lID = localOverlapMap.getLocalElement(gID);
                        targetView(i, j) = sourceView(lID, j);
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
            if(Export_.is_null()){//Export_ not calculated, use inverse of Import_
                target->doExport(*source,*Import_,ADD);
            }
            else{
                target->doExport(*source, *Export_, ADD);
            }
            
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

    template <class SC, class LO, class GO, class NO>
    int Mapper<SC,LO,GO,NO>::insertInto(const XMultiVectorPtr uniqueSource, XMultiVectorPtr & target){
        FROSCH_DETAILTIMER_START(insertInto,"InsertInto");
#if defined(HAVE_XPETRA_KOKKOS_REFACTOR) && defined(HAVE_XPETRA_TPETRA)
            if (target->getMap()->lib() == UseTpetra) {
                ConstXMapPtr uniqueSourceMap = uniqueSource->getMap();
                ConstXMapPtr targetMap = target->getMap();
                auto localUSourceMap = uniqueSourceMap->getLocalMap();
                auto localTargetMap = targetMap->getLocalMap();
                // run local restriction on execution space defined by local-map
                using execution_space = typename XMap::local_map_type::execution_space;
                Kokkos::RangePolicy<execution_space> policy (0, uniqueSourceMap->getNodeNumElements()); //iterate over unqiqueMap
                for (UN i=0; i<target->getNumVectors(); i++) {
                    auto sourceData_i = uniqueSource->getData(i);
                    auto targetLocalData_i = target->getDataNonConst(i);
                    Kokkos::parallel_for(
                      "FROSch_MappingExtern::insertInto", policy,
                      KOKKOS_LAMBDA(const int j) {
                        GO gID = localUSourceMap.getGlobalElement(j);
                        LO lIDTarget = localTargetMap.getLocalElement(gID);
                        targetLocalData_i[lIDTarget] = sourceData_i[j];
                      });
                }
                Kokkos::fence();
            } else {
                exit(1);
            }
#else
    exit(1);
#endif
    return 0;
    }

    //! This method doesn't do any communication betweeen processes!
    //! It inserts all values for entries which are on the own process and also in the transfer Map to the 
    template <class SC, class LO, class GO, class NO>
    int Mapper<SC,LO,GO,NO>::insertIntoWithCheck(const XMultiVectorPtr uniqueSource, XMultiVectorPtr & target){
        FROSCH_DETAILTIMER_START(insertIntoWithCheck,"InsertIntoWithCheck");
#if defined(HAVE_XPETRA_KOKKOS_REFACTOR) && defined(HAVE_XPETRA_TPETRA)
            if (target->getMap()->lib() == UseTpetra) {
                ConstXMapPtr uniqueSourceMap = uniqueSource->getMap();
                ConstXMapPtr targetMap = target->getMap();
                auto localUSourceMap = uniqueSourceMap->getLocalMap();
                auto localTargetMap = targetMap->getLocalMap();
                // run local restriction on execution space defined by local-map
                using execution_space = typename XMap::local_map_type::execution_space;
                Kokkos::RangePolicy<execution_space> policy (0, uniqueSourceMap->getNodeNumElements()); //iterate over unqiqueMap
                for (UN i=0; i<target->getNumVectors(); i++) {
                    auto sourceData_i = uniqueSource->getData(i);
                    auto targetLocalData_i = target->getDataNonConst(i);
                    Kokkos::parallel_for(
                      "FROSch_MappingExtern::insertInto", policy,
                      KOKKOS_LAMBDA(const int j) {
                        GO gID = localUSourceMap.getGlobalElement(j);
                        LO lIDTarget = localTargetMap.getLocalElement(gID);
                        if(lIDTarget>-1){
                            targetLocalData_i[lIDTarget] = sourceData_i[j];
                        }
                      });
                }
                Kokkos::fence();
            } else {
                exit(1);
            }
#else
    exit(1);
#endif
    return 0;
    }

}

#endif