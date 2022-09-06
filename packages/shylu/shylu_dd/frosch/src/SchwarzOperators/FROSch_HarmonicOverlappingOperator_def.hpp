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

#ifndef _FROSCH_HARMONICOVERLAPPINGOPERATOR_DEF_HPP
#define _FROSCH_HARMONICOVERLAPPINGOPERATOR_DEF_HPP

#include <FROSch_HarmonicOverlappingOperator_decl.hpp>

namespace FROSch {

    using namespace std;
    using namespace Teuchos;
    using namespace Xpetra;

    //TODO: Remove
    //#undef NDEBUG

    template <class SC,class LO,class GO,class NO>
    HarmonicOverlappingOperator<SC,LO,GO,NO>::HarmonicOverlappingOperator(ConstXMatrixPtr k,
                                                          ParameterListPtr parameterList) :
    AlgebraicOverlappingOperator<SC,LO,GO,NO> (k,parameterList)
    {
        HarmonicOnOverlap_ = this->ParameterList_->get("HarmonicOnOverlap",false); //Allows to use the debugging output only implemented in this class without using harmonic overlap
        if(HarmonicOnOverlap_){
            FROSCH_ASSERT(this->Combine_ != CombinationType::Averaging, "HarmonicOnOverlap cannot be used with CombinationType==Averaging")
            if(this->Combine_ == CombinationType::Restricted){ //Rasho is implemented more like a fully additive operator internally, only the node sets are "restricted"
                this->Combine_ = CombinationType::Full;         // TODO: Could I alternatively use the restricted export?
                this->Rasho_ = true;
            }
            auto preSolveStrategyString = this->ParameterList_->get("PreSolveStrategy","OnOvlp");
            if (!preSolveStrategyString.compare("OnOverlapping")) {
                PreSolveStrategy_ = PreSolveStrategy::OnOverlapping;
            } else if (!preSolveStrategyString.compare("OnOvlp")) {
                PreSolveStrategy_ = PreSolveStrategy::OnOvlp;
            // } else if (!preSolveStrategyString.compare("OnMultiple")) {
            //     PreSolveStrategy_ = PreSolveStrategy::OnMultiple;
            }
        }
        
    }

    template <class SC,class LO,class GO,class NO>
    int HarmonicOverlappingOperator<SC,LO,GO,NO>::initializeOverlappingOperator(){
        #ifndef NDEBUG
            writeMeta(this->GlobalOverlappingGraph_->getRangeMap());
            output_map(this->GlobalOverlappingGraph_->getRangeMap(),"unique");
            output_map(this->OverlappingMap_,"overlapping");
        #endif

        if(HarmonicOnOverlap_){
            if (this->Multiplicity_.is_null()){
                this->calculateMultiplicity();
            }
            // Create needed maps for the nonoverlapping and overlapping part of the domain, cutNodes only needed for restricted paper
            calculateHarmonicMapsByMultiplicity(this->GlobalOverlappingGraph_, this->Multiplicity_);

            RCP<const Map<LO,GO,NO> > dummy1, dummy2, dummy3, dummy4, dummy5, dummy6;
            calculateHarmonicMaps<SC,LO,GO,NO>(this->GlobalOverlappingGraph_, this->Multiplicity_,
                  dummy1,dummy2,dummy3,dummy4,dummy5,dummy6,
                  this->Rasho_, static_cast<int>(this->PreSolveStrategy_));//for reference output

            
            this->OverlappingMap_=this->LocalSolveMap_;//TODO: The line above could introduce a really weired bug if the overlappingMatrix is constructed with the overlappingMap but now we redefine the map
                                                       // Also the multiplicity needs to calculated before this or could get messes up???
            




            // Create importer between the (non)overlapping part and the extendend domain
            PreSolveMapper_ = rcp(new Mapper<SC,LO,GO,NO>(this->getDomainMap(), PreSolveMap_, PreSolveMap_, PreSolveMap_, this->Multiplicity_, CombinationType::Restricted));
            InnerMapper_ = rcp(new Mapper<SC,LO,GO,NO>(this->getDomainMap(), this->OverlappingMap_, InnerMap_, InnerMap_, this->Multiplicity_, this->Combine_));

            UniqueToInnerMapper_ = rcp(new Mapper<SC,LO,GO,NO>(this->getDomainMap(), InnerMap_, InnerMap_, InnerMap_, this->Multiplicity_, this->Combine_));
            InnerToOverlappingMapper_ = rcp(new Mapper<SC,LO,GO,NO>(InnerMap_, this->OverlappingMap_, this->OverlappingMap_, this->OverlappingMap_, this->Multiplicity_, this->Combine_));
        }
        AlgebraicOverlappingOperator<SC,LO,GO,NO>::initializeOverlappingOperator(); //This is called last, because we use the changed definition of "this->OverlappingMap_"
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int HarmonicOverlappingOperator<SC,LO,GO,NO>::compute(){
        AlgebraicOverlappingOperator<SC,LO,GO,NO>::compute();
        if(HarmonicOnOverlap_){
            setupHarmonicSolver();
        }
        return 0;
    }

    // template <class SC,class LO,class GO,class NO>
    // int HarmonicOverlappingOperator<SC,LO,GO,NO>::updateLocalOverlappingMatrices()
    // {
    //     FROSCH_DETAILTIMER_START_LEVELID(updateLocalOverlappingMatricesTime,"HarmonicOverlappingOperator::updateLocalOverlappingMatrices");
    //     if (this->IsComputed_) { // already computed once and we want to recycle the information. That is why we reset OverlappingMatrix_ to K_, because K_ has been reset at this point
    //         this->OverlappingMatrix_ = this->K_;
    //     }
    //     this->OverlappingMatrix_ = ExtractLocalSubdomainMatrix(this->OverlappingMatrix_,this->OverlappingMap_);
    //     return 0;
    // }

    //! Y = alpha * A^mode * X + beta * Y
    template <class SC,class LO,class GO,class NO>
    void HarmonicOverlappingOperator<SC,LO,GO,NO>::apply(const XMultiVector &x,
                                                 XMultiVector &y,
                                                 bool usePreconditionerOnly,
                                                 ETransp mode,
                                                 SC alpha,
                                                 SC beta) const
    {
        FROSCH_TIMER_START_LEVELID(applyTime,"OverlappingOperator::apply");
        FROSCH_ASSERT(this->IsComputed_,"FROSch::HarmonicOverlappingOperator: HarmonicOverlappingOperator has to be computed before calling apply()");

        static int iteration = 0;

        if (this->YOverlap_.is_null()) {
            this->YOverlap_ = MultiVectorFactory<SC,LO,GO,NO>::Build(this->OverlappingMatrix_->getDomainMap(),x.getNumVectors());
        } else {
            this->YOverlap_->replaceMap(this->OverlappingMatrix_->getDomainMap());
        }
        if (this->XTmp_.is_null()) this->XTmp_ = MultiVectorFactory<SC,LO,GO,NO>::Build(x.getMap(),x.getNumVectors());

        *(this->XTmp_) = x;
        // Apply K first if the framework is not only used as preconditioner: P = M^-1 K
        if (!usePreconditionerOnly && mode == NO_TRANS) { // If mode != NO_TRANS it is applied at the end
            this->K_->apply(x,*(this->XTmp_),mode,ScalarTraits<SC>::one(),ScalarTraits<SC>::zero());
        }
        
        if(HarmonicOnOverlap_){
            FROSCH_ASSERT(W_ != null,"Run preSolve before using this preconditioner");
            // TODO: Define vector of nonoverlappingmap, import in two steps because their global maps dont agree with the maps used for import/export. Restricted mode could help, but is not available in xpetra, just tpetra.
            if (IntermediateInner_.is_null()) {
                IntermediateInner_ = MultiVectorFactory<SC,LO,GO,NO>::Build(InnerMap_,x.getNumVectors());
            } else {
                IntermediateInner_->putScalar(ScalarTraits<SC>::zero());
            }
            // if (this->XOverlap_.is_null()) {
            //     this->XOverlap_ = MultiVectorFactory<SC,LO,GO,NO>::Build(this->OverlappingMap_,x.getNumVectors());
            // }
            // this->Mapper_->setGlobalMap(*(this->XOverlap_));
            // this->Mapper_->insertInto(this->XTmp_, this->XOverlap_);
            // this->Mapper_->setLocalMap(*(this->XOverlap_));
            // - unique -> nonoverlap
            UniqueToInnerMapper_->restrict(this->XTmp_, IntermediateInner_);
            // nonoverlapp -> overlap
            InnerToOverlappingMapper_->restrict(IntermediateInner_,this->XOverlap_);
            #ifndef NDEBUG
            outputWithOtherMap(this->XOverlap_, this->OverlappingMap_, "XOverlap_New", iteration);
            #endif
        } else{
            this->Mapper_->restrict(this->XTmp_, this->XOverlap_); //Restrict into the local overlapping subdomain vector
        }
        #ifndef NDEBUG
        outputWithOtherMap(this->XOverlap_, this->OverlappingMap_, "XOverlap_", iteration); //output fix, because XOverlap is local aver restrict and was never intended for output/export
        #endif
        this->SubdomainSolver_->apply(*(this->XOverlap_),*(this->YOverlap_),mode,ScalarTraits<SC>::one(),ScalarTraits<SC>::zero());
        this->YOverlap_->replaceMap(this->OverlappingMap_);
        this->Mapper_->prolongate(this->YOverlap_, this->XTmp_);

        if (!usePreconditionerOnly && mode != NO_TRANS) {
            this->K_->apply(*(this->XTmp_),*(this->XTmp_),mode,ScalarTraits<SC>::one(),ScalarTraits<SC>::zero());
        }
        y.update(alpha,*(this->XTmp_),beta);
        #ifndef NDEBUG
        auto rcp_x = RCP<const XMultiVector>(&x, false);
        auto rcp_y = RCP<const XMultiVector>(&y, false);
        output(this->XTmp_, "XTmp_", iteration);
        output(this->YOverlap_, "LocalSol", iteration);
        output(rcp_x, "res", iteration);
        output(rcp_y, "sol", iteration);
        #endif
        iteration++;
    }

    template <class SC,class LO,class GO,class NO>
    int HarmonicOverlappingOperator<SC,LO,GO,NO>::setupHarmonicSolver(){
        FROSCH_DETAILTIMER_START_LEVELID(setupHarmonicSolver,"OverlappingOperator::setupHarmonicSolver");
        RCP<FancyOStream> wrappedCout = getFancyOStream (rcpFromRef (std::cout)); // Wrap std::cout in a FancyOStream.
        if (PreSolveStrategy_==PreSolveStrategy::OnOverlapping){
            HarmonicSolver_ = this->SubdomainSolver_; // The preSolve step for "OnOverlapping" uses the same matrix as the apply step, so we can save some work
        } else {
            RCP<Matrix<SC,LO,GO,NO>> ovlpMatrix = ExtractLocalSubdomainMatrixNonConst<SC,LO,GO,NO>(this->K_, PreSolveMap_);
            HarmonicSolver_ = SolverFactory<SC,LO,GO,NO>::Build(ovlpMatrix,
                                                                sublist(this->ParameterList_,"Solver"),
                                                                string("Solver (Level ") + to_string(this->LevelID_) + string(")"));
            HarmonicSolver_->initialize();
            HarmonicSolver_->compute();
        }
        return 0;
    }

    //! Adapt the rhs of the system to make the initial residual compatible with the preconditioner
    template <class SC,class LO,class GO,class NO>
    void HarmonicOverlappingOperator<SC,LO,GO,NO>::preSolve(XMultiVector & rhs){
        FROSCH_DETAILTIMER_START_LEVELID(preSolve,"OverlappingOperator::preSolve");
        FROSCH_ASSERT(this->isComputed(),"Compute preconditioner before starting preSolve routine");
        
        // See "Restricted Additive Schwarz Preconditioners with Harmonic Overlap
        // for Symmetric Positive Definite Linear Systems", (3.5)
        if(HarmonicOnOverlap_){ // TODO: Do i have to calculate g? See formula 3.9 in restricted harmonic
            //Calculate the harmonizing solution W and adapt rhs
            W_ = MultiVectorFactory<SC,LO,GO,NO>::Build(PreSolveMap_,1);
            RhsPreSolveTmp_= MultiVectorFactory<SC,LO,GO,NO>::Build(PreSolveMap_, 1);
            RCP<XMultiVector> rhsRCP = RCP<XMultiVector>(&rhs, false);
            auto Aw = MultiVectorFactory<SC,LO,GO,NO>::Build(rhs.getMap(),1);

            // Import rhs and set dirichlet entries to zero
            switch (PreSolveStrategy_)
            {
            case OnOvlp:
                RhsPreSolveTmp_->doImport(rhs, *(*PreSolveMapper_).Import_, Xpetra::CombineMode::INSERT);//, true
                break;
            case OnOverlapping:
                PreSolveMapper_->insertInto(rhsRCP, RhsPreSolveTmp_);
                break;
            // case OnMultiple:
            //     PreSolveMapper_->insertIntoWithCheck(rhsRCP, RhsPreSolveTmp_); //Cant put rhs values in nonovlp part into multipleMap=Presolvemap, entries dont exist there -> check
            //     break;
            default:
                break;
            }

            PreSolveMapper_->setLocalMap(*RhsPreSolveTmp_);
            PreSolveMapper_->setLocalMap(*W_);
            HarmonicSolver_->apply(*RhsPreSolveTmp_, *W_, NO_TRANS, ScalarTraits<SC>::one(),ScalarTraits<SC>::zero());
            PreSolveMapper_->setGlobalMap(*W_);
            
            // //TODO: Use restricted import, dont sort before?
            // auto tPreM = Xpetra::IO<SC,LO,GO,NO>::Map2TpetraMap(*PreSolveMap_);
            // auto tUnM = Xpetra::IO<SC,LO,GO,NO>::Map2TpetraMap(*this->OverlappingMatrix_->getDomainMap());
            // RCP<FancyOStream> fancy = fancyOStream(rcpFromRef(cout));
            // tPreM->describe(*fancy,VERB_EXTREME);
            // tUnM->describe(*fancy, VERB_EXTREME);

            // cout<<"locally fitted"<<tPreM->isLocallyFitted(*tUnM)<<endl;

            // Bring solution to the full domain: //TODO: Also done in aftersolve later, unify!!! What about lhs vs rhs map? or take map from K or operator?
            switch (PreSolveStrategy_)
            {
            case OnOvlp:
                Aw->doExport(*W_, *(*PreSolveMapper_).Import_, Xpetra::CombineMode::INSERT);//, true, REPLACE?!
                break;
            default:
                Aw->doExport(*W_, *(*PreSolveMapper_).Import_, Xpetra::CombineMode::ADD);//, true, REPLACE?!
                break;
            }
            
            this->K_->apply(*Aw, *Aw);
            #ifndef NDEBUG
            output(RhsPreSolveTmp_,  "rhsPreSolveTmp_",0);
            output(rhsRCP,  "rhs",0);
            #endif
            rhs.update(-1,*Aw,1);//rhs-A*w


            #ifndef NDEBUG
            output(rhsRCP, "rhsHarmonic",0);
            output(W_, "w",0);
            #endif
        }
    }

    //! Adapt the solution of the system to obtain the correct solution,
    //! despite the change in the rhs in the preSolve function
    template <class SC,class LO,class GO,class NO>
    void HarmonicOverlappingOperator<SC,LO,GO,NO>::afterSolve(XMultiVector & lhs){
        FROSCH_DETAILTIMER_START_LEVELID(afterSolve,"OverlappingOperator::afterSolve");
        // Transform the harmonic solution back to the solution of the nonharmonic system using
        // w from Equation 3.5
        if(HarmonicOnOverlap_){
            FROSCH_ASSERT(this->isComputed(),"Compute preconditioner before starting afterSolve routine");
            FROSCH_ASSERT(W_ != null,"Run preSolve before solving the system and running the current function, afterSolve");
            auto wUnique = MultiVectorFactory<SC,LO,GO,NO>::Build(lhs.getMap(),1);

            switch (PreSolveStrategy_)
            {
            case OnOvlp:
                wUnique->doExport(*W_, *(*PreSolveMapper_).Import_, Xpetra::CombineMode::INSERT);//, true, REPLACE?!
                break;
            default:
                wUnique->doExport(*W_, *(*PreSolveMapper_).Import_, Xpetra::CombineMode::ADD);//, true, REPLACE?!
                break;
            }
            lhs.update(1.,*wUnique, 1.);//lhs+W_
        }
    }

    template <class SC,class LO,class GO,class NO>
    string HarmonicOverlappingOperator<SC,LO,GO,NO>::description() const
    {
        return "Harmonic Overlap Operator";
    }

    //TODO: Test if we can work on unique part because we communicate at the end either way.
    template <class SC,class LO,class GO,class NO>
    int HarmonicOverlappingOperator<SC,LO,GO,NO>::calculateHarmonicMapsByMultiplicity(RCP<const CrsGraph<LO,GO,NO> > graph, RCP<MultiVector<SC,LO,GO,NO>> multiplicity)
    {
        // Prepare
        const auto & rowMap = graph->getRowMap();
        const auto & domMap = graph->getColMap();
        const auto & uniqueMap = multiplicity->getMap();

        //Bring the needed multiplicity values on this process
        auto multiplicityExtended = MultiVectorFactory<SC,LO,GO,NO>::Build(domMap,1);
        RCP<Import<LO,GO,NO> >  domImporter = ImportFactory<LO,GO,NO>::Build(uniqueMap,domMap);
        multiplicityExtended->doImport(*multiplicity, *domImporter, Xpetra::CombineMode::INSERT);
        output(multiplicityExtended,"multiplicityExtended",0);

        // Mark interface nodes in rowMap, (we'd like to do this on domMap==colMap)
        auto interfaceNodes = VectorFactory<int,LO,GO,NO>::Build(domMap,1);
        const auto & mult = multiplicityExtended->getData(0);
        // for node in overlapping graph:
        for(LO localRow=0; localRow<rowMap->getNodeNumElements(); localRow++){
            LO local = domMap->getLocalElement(rowMap->getGlobalElement(localRow));
            // if multiplicity>1: O(nodes)
            if(mult[local]>1){
                ArrayView<const LO> neighbours;
                graph->getLocalRowView(localRow, neighbours);
                // for neighbour in graphindices(rownode): O(~6)
                for(LO neighbour : neighbours){
                    if (mult[neighbour]<mult[local]){// neighbour node is adjacent to a region with higher multiplicity overlap
                        //add neighbour to interface nodes
                        //interfaceNodesArray.push_back(domMap->getGlobalElement(neighbour));
                        interfaceNodes->sumIntoGlobalValue(domMap->getGlobalElement(neighbour), 1);
                    }
                }
            }
        }
        // We didnt catch the interface nodes in colMap, so we have to get them from the other domains
        //TODO: This is a problem for overlap=0, why???
        // Communicate back and forth
        auto interfaceNodesUnique = VectorFactory<int,LO,GO,NO>::Build(uniqueMap);
        interfaceNodesUnique->doExport(*interfaceNodes, *domImporter, Xpetra::CombineMode::ADD);
        interfaceNodes->doImport(*interfaceNodesUnique, *domImporter, Xpetra::CombineMode::ADD);
        
        // Calculate nodes
        auto domainNodesArrayView = domMap->getNodeElementList();
        auto innerNodesArray = Teuchos::Array<GO>();//copy domain nodes
        auto ovlpNodesArray = Teuchos::Array<GO>();
        auto restrDomainNodesArray = Teuchos::Array<GO>();
        // Reserve space
        GO nDomain = domMap->getNodeNumElements();
        GO nOvlp = uniqueMap->getNodeNumElements();
        GO nInner = nDomain - nOvlp;
        ovlpNodesArray.reserve(nOvlp);
        innerNodesArray.reserve(nInner);
        restrDomainNodesArray.reserve(nDomain);

        auto interfaceNodesArray = Teuchos::Array<GO>();

        // Calculate all node sets
        const auto & interface = interfaceNodes->getData(0);
        for(GO global : domainNodesArrayView){
            LO local = domMap->getLocalElement(global);
            const bool isOnInterface = interface[local]>0;
            const bool isMultiple = mult[local]>1;
            const bool isNotInOwnUnique = uniqueMap->getLocalElement(global)<0;

            const bool isOvlp = isMultiple && !isOnInterface;//checked!
            const bool isCut = this->Rasho_ && isOnInterface && isNotInOwnUnique; //checked!
            const bool isInner = !isOvlp && !isCut;
            if(isInner) innerNodesArray.push_back(global);
            if(isOvlp) ovlpNodesArray.push_back(global);
            if(!isCut) restrDomainNodesArray.push_back(global);
            if(isOnInterface) interfaceNodesArray.push_back(global);
        }

        //Calculate map
        GO baseIndex = 0;
        RCP<const Comm<LO> > SerialComm = rcp(new MpiComm<LO>(MPI_COMM_SELF));
        RCP<const Comm<LO> > GlobalComm = domMap->getComm();
        RCP<const Map<LO,GO,NO> > ovlpMap = MapFactory<LO,GO,NO>::Build(domMap->lib(),Teuchos::OrdinalTraits<GO>::invalid(), ovlpNodesArray(), baseIndex, GlobalComm);
        RCP<const Map<LO,GO,NO> > innerMap = MapFactory<LO,GO,NO>::Build(domMap->lib(),Teuchos::OrdinalTraits<GO>::invalid(), innerNodesArray(), baseIndex, GlobalComm);
        RCP<const Map<LO,GO,NO> > restrDomainMap = MapFactory<LO,GO,NO>::Build(domMap->lib(),Teuchos::OrdinalTraits<GO>::invalid(), restrDomainNodesArray(), baseIndex, GlobalComm);

        RCP<const Map<LO,GO,NO> > interfaceMap = MapFactory<LO,GO,NO>::Build(domMap->lib(),Teuchos::OrdinalTraits<GO>::invalid(), interfaceNodesArray(), baseIndex, GlobalComm);
        
        // Assign the maps
        this->InnerMap_ = innerMap;
        RCP<const Map<LO,GO,NO> > newDomainMap = this->Rasho_ ? restrDomainMap : this->OverlappingMap_;
        switch (PreSolveStrategy_){
            case OnOvlp:
                this->PreSolveMap_ = ovlpMap;
                this->LocalSolveMap_ = newDomainMap;
                break;
            case OnOverlapping:
                this->PreSolveMap_ =  newDomainMap;
                this->LocalSolveMap_ = newDomainMap;
                break;
            default:
                exit(1);
                break;
        }

        //TODO: Remove
        bool out_maps=true;
        // #ifndef NDEBUG
        if(out_maps){
            output_map(ovlpMap,"ovlp");
            output_map(innerMap,"inner");
            output_map(restrDomainMap, "restrDomain");

            output_map(interfaceMap, "interface");
        }// #endif

        return 0;
    }
}

#endif