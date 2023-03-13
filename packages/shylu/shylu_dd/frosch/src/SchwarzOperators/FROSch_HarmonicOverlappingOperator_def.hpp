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
#include <FROSch_Tools_def.hpp>

namespace FROSch {

    using namespace std;
    using namespace Teuchos;
    using namespace Xpetra;

    template <class SC,class LO,class GO,class NO>
    HarmonicOverlappingOperator<SC,LO,GO,NO>::HarmonicOverlappingOperator(ConstXMatrixPtr k,
                                                          ParameterListPtr parameterList) :
    AlgebraicOverlappingOperator<SC,LO,GO,NO> (k,parameterList)
    {
        HarmonicOnOverlap_ = this->ParameterList_->get("HarmonicOnOverlap",false); //Allows to use the debugging output only implemented in this class without using harmonic overlap
        if(HarmonicOnOverlap_){
            FROSCH_ASSERT(this->Combine_ != OverlapCombinationType::Averaging, "HarmonicOnOverlap cannot be used with OverlapCombinationType==Averaging")
            if(this->Combine_ == OverlapCombinationType::Restricted){ //Rasho is implemented more like a fully additive operator internally, only the node sets are "restricted"
                this->Combine_ = OverlapCombinationType::Full;
                this->Rasho_ = true;
            }
            auto preSolveStrategyString = this->ParameterList_->get("PreSolveStrategy","OnOvlp");
            if (!preSolveStrategyString.compare("OnOverlapping")) {
                PreSolveStrategy_ = PreSolveStrategy::OnOverlapping;
            } else if (!preSolveStrategyString.compare("OnOvlp")) {
                PreSolveStrategy_ = PreSolveStrategy::OnOvlp;
            } else{
                TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid value for PreSolveStrategy_ read in from Parameter file: " << preSolveStrategyString);
            }

            auto calcInterfaceStrategyString = this->ParameterList_->get("CalcInterfaceStrategy","Exact");
            if (!calcInterfaceStrategyString.compare("Exact")) {
                CalcInterfaceStrategy_ = CalcInterfaceStrategy::Exact;
            } else if (!calcInterfaceStrategyString.compare("ByMultiplicity")) {
                CalcInterfaceStrategy_ = CalcInterfaceStrategy::ByMultiplicity;
            } else if (!calcInterfaceStrategyString.compare("ByRhsHarmonic")) {
                CalcInterfaceStrategy_ = CalcInterfaceStrategy::ByRhsHarmonic;
            }else{
                TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid value for CalcInterfaceStrategy_ read in from Parameter file: " << calcInterfaceStrategyString);
            }
        }        
    }

    template <class SC,class LO,class GO,class NO>
    int HarmonicOverlappingOperator<SC,LO,GO,NO>::initializeOverlappingOperator(){
        // TODO: Is this always "true"?
        UniqueMap_ = this->getDomainMap();

        if(OutputMapsAndVectors_){
            writeMeta(this->GlobalOverlappingGraph_->getRangeMap());
            output_map(this->GlobalOverlappingGraph_->getRangeMap(),"unique");
            output_map(this->OverlappingMap_,"overlapping");
        } 

        if(HarmonicOnOverlap_){
            if (this->Multiplicity_.is_null()){
                this->calculateMultiplicity();
            }
            // Create needed maps for the nonoverlapping and overlapping part of the domain, cutNodes only needed for restricted paper
            calculateHarmonicMaps();

            this->OverlappingMap_=this->LocalSolveMap_;//TODO: The line above could introduce really weired bugs
                                                       // e. g. if the overlappingMatrix is constructed with the overlappingMap but now we redefine the map
                                                       // Also the multiplicity needs to calculated before this or could get messes up???
            
            PreSolveMapper_ = rcp(new Mapper<SC,LO,GO,NO>(UniqueMap_, PreSolveMap_, PreSolveMap_, PreSolveMap_, this->Multiplicity_, OverlapCombinationType::Restricted));
            ResidualMapper_ = rcp(new Mapper<SC,LO,GO,NO>(UniqueMap_, this->OverlappingMap_, ResidualMap_, ResidualMap_, this->Multiplicity_, this->Combine_));

            UniqueToResidualMapper_ = rcp(new Mapper<SC,LO,GO,NO>(UniqueMap_, ResidualMap_, ResidualMap_, ResidualMap_, this->Multiplicity_, this->Combine_));
            ResidualToOverlappingMapper_ = rcp(new Mapper<SC,LO,GO,NO>(ResidualMap_, this->OverlappingMap_, this->OverlappingMap_, this->OverlappingMap_, this->Multiplicity_, this->Combine_));
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
            if (IntermediateResidual_.is_null()) {
                IntermediateResidual_ = MultiVectorFactory<SC,LO,GO,NO>::Build(ResidualMap_,x.getNumVectors());
            } else {
                IntermediateResidual_->putScalar(ScalarTraits<SC>::zero());
            }
            // if (this->XOverlap_.is_null()) {
            //     this->XOverlap_ = MultiVectorFactory<SC,LO,GO,NO>::Build(this->OverlappingMap_,x.getNumVectors());
            // }
            // this->Mapper_->setGlobalMap(*(this->XOverlap_));
            // this->Mapper_->insertInto(this->XTmp_, this->XOverlap_);
            // this->Mapper_->setLocalMap(*(this->XOverlap_));
            // - unique -> nonoverlap
            UniqueToResidualMapper_->restrict(this->XTmp_, IntermediateResidual_);
            // nonoverlapp -> overlap
            ResidualToOverlappingMapper_->restrict(IntermediateResidual_,this->XOverlap_);
            if(OutputMapsAndVectors_){
                outputWithOtherMap(this->XOverlap_, this->OverlappingMap_, "XOverlap_New", iteration);
            }
        } else{
            this->Mapper_->restrict(this->XTmp_, this->XOverlap_); //Restrict into the local overlapping subdomain vector
        }
        if(OutputMapsAndVectors_){
            outputWithOtherMap(this->XOverlap_, this->OverlappingMap_, "XOverlap_", iteration); //output fix, because XOverlap is local aver restrict and was never intended for output/export
        }
        this->SubdomainSolver_->apply(*(this->XOverlap_),*(this->YOverlap_),mode,ScalarTraits<SC>::one(),ScalarTraits<SC>::zero());
        this->YOverlap_->replaceMap(this->OverlappingMap_);
        this->Mapper_->prolongate(this->YOverlap_, this->XTmp_);

        if (!usePreconditionerOnly && mode != NO_TRANS) {
            this->K_->apply(*(this->XTmp_),*(this->XTmp_),mode,ScalarTraits<SC>::one(),ScalarTraits<SC>::zero());
        }
        y.update(alpha,*(this->XTmp_),beta);
        if(OutputMapsAndVectors_){
            auto rcp_x = RCP<const XMultiVector>(&x, false);
            auto rcp_y = RCP<const XMultiVector>(&y, false);
            output(this->XTmp_, "XTmp_", iteration);
            output(this->YOverlap_, "LocalSol", iteration);
            output(rcp_x, "res", iteration);
            output(rcp_y, "sol", iteration);
        }
        iteration++;
    }

    template <class SC,class LO,class GO,class NO>
    int HarmonicOverlappingOperator<SC,LO,GO,NO>::setupHarmonicSolver(){
        FROSCH_DETAILTIMER_START_LEVELID(setupHarmonicSolver,"OverlappingOperator::setupHarmonicSolver");
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

        if(HarmonicOnOverlap_){
            //Calculate the harmonizing solution W and adapt rhs
            auto WLocal = MultiVectorFactory<SC,LO,GO,NO>::Build(PreSolveMap_,1);
            W_ = MultiVectorFactory<SC,LO,GO,NO>::Build(UniqueMap_,1);
            XMultiVectorPtr RhsPreSolveTmp_= MultiVectorFactory<SC,LO,GO,NO>::Build(PreSolveMap_, 1);
            RCP<XMultiVector> rhsRCP = RCP<XMultiVector>(&rhs, false);
            auto Aw = MultiVectorFactory<SC,LO,GO,NO>::Build(rhs.getMap(),1);

            // Import rhs
            switch (PreSolveStrategy_)
            {
            case OnOvlp:
                RhsPreSolveTmp_->doImport(rhs, *(*PreSolveMapper_).Import_, Xpetra::CombineMode::INSERT);//, true
                break;
            case OnOverlapping:
                PreSolveMapper_->insertInto(rhsRCP, RhsPreSolveTmp_);
                break;
            default:
                TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Invalid value for PreSolveStrategy_ occurred: " << PreSolveStrategy_);
                break;
            }

            PreSolveMapper_->setLocalMap(*RhsPreSolveTmp_);
            PreSolveMapper_->setLocalMap(*WLocal);
            HarmonicSolver_->apply(*RhsPreSolveTmp_, *WLocal, NO_TRANS, ScalarTraits<SC>::one(),ScalarTraits<SC>::zero());
            PreSolveMapper_->setGlobalMap(*WLocal);

            // Bring solution to the full domain:
            switch (PreSolveStrategy_)
            {
            case OnOvlp:
                W_->doExport(*WLocal, *(*PreSolveMapper_).Import_, Xpetra::CombineMode::INSERT);
                break;
            default:
                W_->doExport(*WLocal, *(*PreSolveMapper_).Import_, Xpetra::CombineMode::ADD);
                break;
            }
            
            this->K_->apply(*W_, *Aw);
            if(OutputMapsAndVectors_){
                output(RhsPreSolveTmp_,  "rhsPreSolveTmp_",0);
                output(rhsRCP,  "rhs",0);
            }
            rhs.update(-1,*Aw,1);//rhs-A*w
            //TODO: Save RhsHarmonic_

            if(OutputMapsAndVectors_){
                output(rhsRCP, "rhsHarmonic",0);
                output(W_, "w",0);
            }
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
            lhs.update(1.,*W_, 1.);//lhs+W_
        }

        if(OutputMapsAndVectors_){
                RCP<XMultiVector> lhsRCP = RCP<XMultiVector>(&lhs, false);
                output(lhsRCP, "solution_final",0);
            }
    }

    template <class SC,class LO,class GO,class NO>
    string HarmonicOverlappingOperator<SC,LO,GO,NO>::description() const
    {
        return "Harmonic Overlap Operator";
    }

    template <class SC,class LO,class GO,class NO>
    int HarmonicOverlappingOperator<SC,LO,GO,NO>::assignMaps(ConstXMapPtr interfaceMap, ConstXMapPtr ovlpMap, ConstXMapPtr innerMap, ConstXMapPtr restrDomainMap)
    {
        // Assign the maps
        RCP<const Map<LO,GO,NO> > newDomainMap = this->Rasho_ ? restrDomainMap : this->OverlappingMap_;
        switch (PreSolveStrategy_){
            case OnOvlp:
                this->PreSolveMap_      = ovlpMap;
                this->ResidualMap_      = innerMap;
                this->LocalSolveMap_    = newDomainMap;
                break;
            case OnOverlapping:
                this->PreSolveMap_      = newDomainMap;
                this->ResidualMap_      = interfaceMap;
                this->LocalSolveMap_    = newDomainMap;
                break;
            default:
                TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Invalid value for PreSolveStrategy_ occurred: " << PreSolveStrategy_);
                break;
        }
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    RCP<MultiVector<int,LO,GO,NO>> HarmonicOverlappingOperator<SC,LO,GO,NO>::calculateInterfaceExact(){
        FROSCH_DETAILTIMER_START(calculateInterfaceExactTime,"calculateInterfaceExact");
        auto graph = this->GlobalOverlappingGraph_;
        RCP<const Map<LO,GO,NO> > dummyOutputMap =MapFactory<LO,GO,NO>::Build(graph->getColMap(),1);
        ExtendOverlapByOneLayer(graph, dummyOutputMap,graph, dummyOutputMap);

        auto colMap = graph->getColMap(); // Result different from previous call!
        auto rowMap = graph->getRowMap();

        auto interfaceVectorCol = MultiVectorFactory<int,LO,GO,NO>::Build(colMap,1);
        auto interfaceVector = MultiVectorFactory<int,LO,GO,NO>::Build(rowMap,1);
        for(auto & globalIndex: colMap->getNodeElementList()){
            bool inRowMap = rowMap->isNodeGlobalElement(globalIndex);
            if(!inRowMap){
               interfaceVectorCol->sumIntoGlobalValue(globalIndex,0, 1);
            }
        }
        // Communicate back and forth
        auto colImporter = ImportFactory<LO,GO,NO>::Build(UniqueMap_,colMap);
        auto rowImporter = ImportFactory<LO,GO,NO>::Build(UniqueMap_,rowMap);
        auto interfaceVectorUnique = VectorFactory<int,LO,GO,NO>::Build(UniqueMap_);
        interfaceVectorUnique->doExport(*interfaceVectorCol, *colImporter, Xpetra::CombineMode::ADD);
        interfaceVector->doImport(*interfaceVectorUnique, *rowImporter, Xpetra::CombineMode::ADD);
        return interfaceVector;
    }

    template <class SC,class LO,class GO,class NO>
    RCP<MultiVector<int,LO,GO,NO>> HarmonicOverlappingOperator<SC,LO,GO,NO>::calculateInterfaceByRhsHarmonic(){
        FROSCH_DETAILTIMER_START(calculateInterfaceByRhsHarmonicTime,"calculateInterfaceByRhsHarmonic");
        auto graph = this->GlobalOverlappingGraph_;
        auto domMap = graph->getColMap();

        auto importer = ImportFactory<LO,GO,NO>::Build(RhsHarmonic_->getMap(),domMap);
        auto rhsHarmonicOverlapping = MultiVectorFactory<SC,LO,GO,NO>::Build(domMap,1);
        rhsHarmonicOverlapping->doImport(*RhsHarmonic_, *importer, Xpetra::CombineMode::INSERT); //Bring the needed rhsHarmonic values on this process

        // TODO: Problem: We may not do this with a rhs where some parts are zero from the beginning, but can change
        // with another execution. Then we miss nodes and the algorithm doesnt converge
        SC eps = 1e-10;//TODO: what to choose as epsilon?
        const auto & rhsH = rhsHarmonicOverlapping->getData(0);
        auto interfaceVector = MultiVectorFactory<int,LO,GO,NO>::Build(domMap,1);
        for (size_t local=0; local<domMap->getNodeNumElements(); local++){
            if(rhsH[local]>eps){ // If the rhs is > eps, the alg. failed to set the rhs to zero there -> need to import residum there.
                auto globalIndex = domMap->getGlobalElement(local);
                interfaceVector->sumIntoGlobalValue(globalIndex, 0,1);
            }
        }
        //Don't need to communicate back and forth
        return interfaceVector;
    }

    template <class SC,class LO,class GO,class NO>
    RCP<MultiVector<int,LO,GO,NO>> HarmonicOverlappingOperator<SC,LO,GO,NO>::calculateInterfaceByMultiplicity()
    {
        FROSCH_DETAILTIMER_START(calculateInterfaceByMultiplicityTime,"calculateInterfaceByMultiplicity");

        auto graph = this->GlobalOverlappingGraph_;
        auto multiplicity = this->Multiplicity_;
        // Prepare
        const auto & rowMap = graph->getRowMap();
        const auto & domMap = graph->getColMap();

        //Bring the needed multiplicity values on this process
        auto multiplicityExtended = MultiVectorFactory<SC,LO,GO,NO>::Build(domMap,1);
        RCP<Import<LO,GO,NO> >  domImporter = ImportFactory<LO,GO,NO>::Build(UniqueMap_,domMap);
        multiplicityExtended->doImport(*multiplicity, *domImporter, Xpetra::CombineMode::INSERT);
        if(OutputMapsAndVectors_){
            output(multiplicityExtended,"multiplicityExtended",0);
        }
        // Mark interface nodes in rowMap, (we'd like to do this on domMap==colMap)
        auto interfaceVector = MultiVectorFactory<int,LO,GO,NO>::Build(domMap,1);
        const auto & mult = multiplicityExtended->getData(0);

        for(size_t localRow=0; localRow<rowMap->getNodeNumElements(); localRow++){// for node in overlapping graph:
            LO local = domMap->getLocalElement(rowMap->getGlobalElement(localRow));
            if(mult[local]>1){// if multiplicity>1: O(nodes)
                ArrayView<const LO> neighbours;
                graph->getLocalRowView(localRow, neighbours);
                for(LO neighbour : neighbours){ // for neighbour in graphindices(rownode): O(~6)
                    if (mult[neighbour]<mult[local]){ // neighbour node is adjacent to a region with higher multiplicity overlap
                        interfaceVector->sumIntoGlobalValue(domMap->getGlobalElement(neighbour), 0,1); //add neighbour to interface nodes
                    }
                }
            }
        }
        // Communicate back and forth
        auto interfaceVectorUnique = MultiVectorFactory<int,LO,GO,NO>::Build(UniqueMap_,1);
        interfaceVectorUnique->doExport(*interfaceVector, *domImporter, Xpetra::CombineMode::ADD);
        interfaceVector->doImport(*interfaceVectorUnique, *domImporter, Xpetra::CombineMode::ADD);
        return interfaceVector;
    }

    template <class SC,class LO,class GO,class NO>
    int HarmonicOverlappingOperator<SC,LO,GO,NO>::calculateHarmonicMaps()
    {
        switch (CalcInterfaceStrategy_){
            case Exact:
                Interfaces_ = calculateInterfaceExact();
                break;
            case ByMultiplicity:
                Interfaces_ = calculateInterfaceByMultiplicity();
                break;
            case ByRhsHarmonic:
                TEUCHOS_TEST_FOR_EXCEPTION(!(PreSolveStrategy_==OnOverlapping), std::invalid_argument, "CalculateInterface::ByRhsHarmonic can only be combined with PreSolveStrategy::OnOverlapping");
                TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented!");
                // Would need to call calcHarmonicMaps after PreSolve with rhs and then set up the residualMappers and in case of rasho set up a new solver or edit the current one (and recompute)
                break;
            default:
                TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Invalid value for CalcInterfaceStrategy_ occurred: " << CalcInterfaceStrategy_);
                break;
        }
        if(OutputMapsAndVectors_){
            output(Interfaces_,"interfaceVector",0);
        }
        return calculateHarmonicMapsFromInterface();
        
    }

    template <class SC,class LO,class GO,class NO>
    int HarmonicOverlappingOperator<SC,LO,GO,NO>::calculateHarmonicMapsFromInterface()
    {
        FROSCH_DETAILTIMER_START(calculateHarmonicMaps,"CalculateHarmonicMaps");

        auto graph = this->GlobalOverlappingGraph_;
        const auto & domMap = graph->getColMap();

        auto multiplicityExtended = MultiVectorFactory<SC,LO,GO,NO>::Build(domMap,1);
        RCP<Import<LO,GO,NO> >  domImporter = ImportFactory<LO,GO,NO>::Build(UniqueMap_,domMap);
        multiplicityExtended->doImport(*(this->Multiplicity_), *domImporter, Xpetra::CombineMode::INSERT);

        // Calculate nodes
        auto domainNodesArrayView = domMap->getNodeElementList(); //copy domain nodes
        auto restrInnerNodesArray = Teuchos::Array<GO>();
        auto ovlpNodesArray = Teuchos::Array<GO>();
        auto restrDomainNodesArray = Teuchos::Array<GO>();
        auto restrInterfaceNodesArray = Teuchos::Array<GO>();
        
        // Reserve space
        GO nDomain = domMap->getNodeNumElements();
        GO nOvlp = UniqueMap_->getNodeNumElements();
        GO nInner = nDomain - nOvlp;
        ovlpNodesArray.reserve(nOvlp);
        restrInnerNodesArray.reserve(nInner);
        restrDomainNodesArray.reserve(nDomain);

        const bool rasho = this->Rasho_;
        // Calculate all node sets
        const auto & interface = Interfaces_->getData(0);
        const auto & mult = multiplicityExtended->getData(0);
        for(GO global : domainNodesArrayView){
            LO local = domMap->getLocalElement(global);
            
            const bool isOnInterface = interface[local]>0;
            const bool isMultiple = mult[local]>1;
            const bool isNotInOwnUnique = UniqueMap_->getLocalElement(global)<0;

            const bool isOvlp = isMultiple && !isOnInterface;//checked!
            const bool isCut = rasho && isOnInterface && isNotInOwnUnique; //checked!
            const bool isInner = !isOvlp && !isCut;
            if(isInner) restrInnerNodesArray.push_back(global);
            if(isOvlp) ovlpNodesArray.push_back(global);
            if(!isCut) restrDomainNodesArray.push_back(global);
            if(isOnInterface && !isCut) restrInterfaceNodesArray.push_back(global);
        }      
        //Calculate map
        GO baseIndex = 0;
        //RCP<const Comm<LO> > SerialComm = rcp(new MpiComm<LO>(MPI_COMM_SELF));
        RCP<const Comm<LO> > GlobalComm = domMap->getComm();
        RCP<const Map<LO,GO,NO> > ovlpMap = MapFactory<LO,GO,NO>::Build(domMap->lib(),Teuchos::OrdinalTraits<GO>::invalid(), ovlpNodesArray(), baseIndex, GlobalComm);
        RCP<const Map<LO,GO,NO> > innerMap = MapFactory<LO,GO,NO>::Build(domMap->lib(),Teuchos::OrdinalTraits<GO>::invalid(), restrInnerNodesArray(), baseIndex, GlobalComm);
        RCP<const Map<LO,GO,NO> > restrDomainMap = MapFactory<LO,GO,NO>::Build(domMap->lib(),Teuchos::OrdinalTraits<GO>::invalid(), restrDomainNodesArray(), baseIndex, GlobalComm);
        RCP<const Map<LO,GO,NO> > interfaceMap = MapFactory<LO,GO,NO>::Build(domMap->lib(),Teuchos::OrdinalTraits<GO>::invalid(), restrInterfaceNodesArray(), baseIndex, GlobalComm);
        
        assignMaps(interfaceMap, ovlpMap, innerMap, restrDomainMap);

        if(OutputMapsAndVectors_){
            output_map(ovlpMap,"ovlp");
            output_map(innerMap,"inner");
            output_map(restrDomainMap, "restrDomain");
            output_map(interfaceMap, "interface");
        }

        return 0;  
    }

    template <class SC,class LO,class GO,class NO>
    void HarmonicOverlappingOperator<SC,LO,GO,NO>::printParameterDescription() const
    {
        AlgebraicOverlappingOperator<SC,LO,GO,NO>::printParameterDescription();
        cout
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << setw(89) << "========================================================================================="
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(41) << "HarmonicOnOverlap" << right
            << " | " << setw(41) << this->ParameterList_->get("HarmonicOnOverlap",false)
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(41) << "PreSolveStrategy" << right
            << " | " << setw(41) << this->ParameterList_->get("PreSolveStrategy","OnOvlp")
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(41) << "CalcInterfaceStrategy" << right
            << " | " << setw(41) << this->ParameterList_->get("CalcInterfaceStrategy","Exact")
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(41) << "Rasho" << right
            << " | " << setw(41) << this->Rasho_
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << setw(89) << "-----------------------------------------------------------------------------------------"
            << endl;
    }
}

#endif