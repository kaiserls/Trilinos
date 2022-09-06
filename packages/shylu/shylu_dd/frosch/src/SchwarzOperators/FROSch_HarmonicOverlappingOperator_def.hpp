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

        // TODO: Is this always "true"?
        UniqueMap_ = this->getDomainMap();
        
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
            calculateHarmonicMaps();

            // RCP<const Map<LO,GO,NO> > dummy1, dummy2, dummy3, dummy4, dummy5, dummy6;
            // calculateHarmonicMaps(this->GlobalOverlappingGraph_, this->Multiplicity_,
            //       dummy1,dummy2,dummy3,dummy4,dummy5,dummy6,
            //       this->Rasho_, static_cast<int>(this->PreSolveStrategy_));//for reference output

            
            this->OverlappingMap_=this->LocalSolveMap_;//TODO: The line above could introduce a really weired bug if the overlappingMatrix is constructed with the overlappingMap but now we redefine the map
                                                       // Also the multiplicity needs to calculated before this or could get messes up???
            




            // Create importer between the (non)overlapping part and the extendend domain
            PreSolveMapper_ = rcp(new Mapper<SC,LO,GO,NO>(UniqueMap_, PreSolveMap_, PreSolveMap_, PreSolveMap_, this->Multiplicity_, CombinationType::Restricted));
            ResidualMapper_ = rcp(new Mapper<SC,LO,GO,NO>(UniqueMap_, this->OverlappingMap_, ResidualMap_, ResidualMap_, this->Multiplicity_, this->Combine_));

            UniqueToResidualMapper_ = rcp(new Mapper<SC,LO,GO,NO>(UniqueMap_, ResidualMap_, ResidualMap_, ResidualMap_, this->Multiplicity_, this->Combine_));
            InnerToOverlappingMapper_ = rcp(new Mapper<SC,LO,GO,NO>(ResidualMap_, this->OverlappingMap_, this->OverlappingMap_, this->OverlappingMap_, this->Multiplicity_, this->Combine_));
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
                IntermediateInner_ = MultiVectorFactory<SC,LO,GO,NO>::Build(ResidualMap_,x.getNumVectors());
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
            UniqueToResidualMapper_->restrict(this->XTmp_, IntermediateInner_);
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
            std::cout<<"before local";
            //Calculate the harmonizing solution W and adapt rhs
            auto WLocal = MultiVectorFactory<SC,LO,GO,NO>::Build(PreSolveMap_,1);
            W_ = MultiVectorFactory<SC,LO,GO,NO>::Build(UniqueMap_,1);
            std::cout<<"before rhspresolve";
            XMultiVectorPtr RhsPreSolveTmp_= MultiVectorFactory<SC,LO,GO,NO>::Build(PreSolveMap_, 1);
            RCP<XMultiVector> rhsRCP = RCP<XMultiVector>(&rhs, false);
            auto Aw = MultiVectorFactory<SC,LO,GO,NO>::Build(rhs.getMap(),1);

            std::cout<<"before isnert";

            // Import rhs and set dirichlet entries to zero
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

            std::cout<<"reached begind insert";

            PreSolveMapper_->setLocalMap(*RhsPreSolveTmp_);
            PreSolveMapper_->setLocalMap(*WLocal);
            HarmonicSolver_->apply(*RhsPreSolveTmp_, *WLocal, NO_TRANS, ScalarTraits<SC>::one(),ScalarTraits<SC>::zero());
            PreSolveMapper_->setGlobalMap(*W_);
            
            // //TODO: Use restricted import, dont sort before?
            // auto tPreM = Xpetra::IO<SC,LO,GO,NO>::Map2TpetraMap(*PreSolveMap_);
            // auto tUnM = Xpetra::IO<SC,LO,GO,NO>::Map2TpetraMap(*this->OverlappingMatrix_->getDomainMap());
            // RCP<FancyOStream> fancy = fancyOStream(rcpFromRef(cout));
            // tPreM->describe(*fancy,VERB_EXTREME);
            // tUnM->describe(*fancy, VERB_EXTREME);

            // cout<<"locally fitted"<<tPreM->isLocallyFitted(*tUnM)<<endl;

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
            lhs.update(1.,*W_, 1.);//lhs+W_
        }
    }

    template <class SC,class LO,class GO,class NO>
    string HarmonicOverlappingOperator<SC,LO,GO,NO>::description() const
    {
        return "Harmonic Overlap Operator";
    }

    template <class SC,class LO,class GO,class NO>
    int HarmonicOverlappingOperator<SC,LO,GO,NO>::calculateHarmonicMaps()
    {
        switch (CalcInterfaceStrategy_){
            case Exact:
                calculateHarmonicMapsExact();
                break;
            case ByMultiplicity:
                calculateHarmonicMapsByMultiplicity();
                break;
            case ByRhsHarmonic:
                TEUCHOS_TEST_FOR_EXCEPTION(PreSolveStrategy_==OnOverlapping, std::invalid_argument, "CalculateInterface::ByRhsHarmonic can only be combined with PreSolveStrategy::OnOverlapping");
                calculateHarmonicMapsByRhsHarmonic();
                break;
            default:
                TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Invalid value for CalcInterfaceStrategy_ occurred: " << CalcInterfaceStrategy_);
                break;
        }
        return 0;
    }

    //TODO: Make interfaceMap without cutNodes in the calcHarmMap methods!
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
    int HarmonicOverlappingOperator<SC,LO,GO,NO>::calculateHarmonicMapsByMultiplicity()
    {
        FROSCH_DETAILTIMER_START(calculateHarmonicMapsByMultiplicityTime,"CalculateHarmonicMapsByMultiplicity");
        auto graph = this->GlobalOverlappingGraph_;
        auto multiplicity = this->Multiplicity_;
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
        //TODO: This is a problem for overlap~<3
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
        
        assignMaps(interfaceMap, ovlpMap, innerMap, restrDomainMap);
        
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

    //! This function calculates all the maps associated with the harmonic overlapping operator.
    //! Some are not actually needed in the implementation but can be used for debugging and comparison with papers
    template <class SC,class LO,class GO,class NO>
    int HarmonicOverlappingOperator<SC,LO,GO,NO>::calculateHarmonicMapsExact(){
        FROSCH_DETAILTIMER_START(calculateHarmonicMapsExactTime,"CalculateHarmonicMapsExact");

        auto graph = this->GlobalOverlappingGraph_;
        auto multiplicity = this->Multiplicity_;

        //The input graph includes all nodes belonging to the 
        RCP<const Map<LO,GO,NO> > dummyOutputMap =MapFactory<LO,GO,NO>::Build(graph->getColMap(),1);
        ExtendOverlapByOneLayer(graph, dummyOutputMap,graph, dummyOutputMap);
        
        const auto & rowMap = graph->getRowMap();
        const auto & colMap = graph->getColMap();
        const auto & uniqueMap = multiplicity->getMap();
        
        //Bring the needed multiplicity values on this process
        auto multiplicityExtended = MultiVectorFactory<SC,LO,GO,NO>::Build(colMap,1);//Each process has to know the multiplicity of their own nodes.
                                                                              //The interfaces are recognized through the graph, not the multiplicity
        RCP<Import<LO,GO,NO> >  colImporter = ImportFactory<LO,GO,NO>::Build(uniqueMap,colMap);
        RCP<Import<LO,GO,NO> >  rowImporter = ImportFactory<LO,GO,NO>::Build(uniqueMap,rowMap);
        multiplicityExtended->doImport(*multiplicity, *colImporter, Xpetra::CombineMode::INSERT);

        //TODO: Remove because this is just output to test something, not further used in calculation:
        auto multiplicityRow = MultiVectorFactory<SC,LO,GO,NO>::Build(rowMap,1);
        auto multiplicityRowExtended = MultiVectorFactory<SC,LO,GO,NO>::Build(colMap,1);
        multiplicityRow->doImport(*multiplicity, *rowImporter, Xpetra::CombineMode::INSERT);
        #ifndef NDEBUG   
        output(multiplicityRow,"multRow");
        output(multiplicityExtended, "multCol");
        #endif

        //Bring the needed interface nodes onto this process
        auto interfaceNodes = getGlobalInterfaceNodes<LO,GO,NO>(graph);
        #ifndef NDEBUG
        output(interfaceNodes, "interfaceEncoded");
        #endif
        // Communicate over unique map to same map again (or mpi call) to ensure that values are exchanged
        auto interfaceNodesUnique = VectorFactory<int,LO,GO,NO>::Build(uniqueMap);
        interfaceNodesUnique->doExport(*interfaceNodes, *colImporter, Xpetra::CombineMode::ADD);
        interfaceNodes->doImport(*interfaceNodesUnique, *colImporter, Xpetra::CombineMode::INSERT);

        // Calculate nodes
        auto domainNodesArrayView = rowMap->getNodeElementList();//rowMap so we don't get the interface of the "extended" domain
        auto innerNodesArray = Teuchos::Array<GO>();//copy domain nodes
        auto interfaceNodesArray = Teuchos::Array<GO>();//empty array
        auto overlapNodesArray = Teuchos::Array<GO>();//empty array
        auto cutNodesArray = Teuchos::Array<GO>();//empty array
        Teuchos::Array<GO> matrixImportArray; // assigned to later

        // Reserve space
        GO nDomain = colMap->getNodeNumElements();
        GO nOvlp = uniqueMap->getNodeNumElements();
        GO nInner = nDomain - nOvlp;
        overlapNodesArray.reserve(nOvlp);
        innerNodesArray.reserve(nInner);
        interfaceNodesArray.reserve(4*sqrt(nDomain));

        const auto & mult = multiplicityExtended->getData(0);
        const auto & interface = interfaceNodes->getData(0);

        // InterfaceNodes, OvlpNodes and CutNodes, InnerNodes and MultipleNodes
        for(GO global : domainNodesArrayView){
            LO local = colMap->getLocalElement(global);
            bool isOnInterface = interface[local]>0;
            bool isMultiple = mult[local]>1;
            bool isNotInOwnUnique = uniqueMap->getLocalElement(global)<0;
            if(isMultiple && !isOnInterface) overlapNodesArray.push_back(global);
            if(isOnInterface) interfaceNodesArray.push_back(global);
            if(isOnInterface && isNotInOwnUnique) cutNodesArray.push_back(global);
            if(!isMultiple || isOnInterface) innerNodesArray.push_back(global);
        }

        if(Rasho_){//could be much faster, already computed cutNodesArray, only a few elements
            matrixImportArray = Teuchos::Array<GO>(domainNodesArrayView);
            // auto remove_cut_nodes_condition = [colMap, uniqueMap, interface](const GO& node) {
            //     return uniqueMap->getLocalElement(global)<0 && interface[colMap->getLocalElement(node)]>0; // erase if on in the cut nodes set
            matrixImportArray.erase(std::remove_if(matrixImportArray.begin(), matrixImportArray.end(),
                [colMap, uniqueMap, interface](const GO& node) {
                    return uniqueMap->getLocalElement(node)<0 && interface[colMap->getLocalElement(node)]>0; // erase if on in the cut nodes set
                    }
                ), matrixImportArray.end());
        } else {
            matrixImportArray = Teuchos::Array<GO>(domainNodesArrayView);
        }

        //Calculate maps
        GO baseIndex = 0;
        RCP<const Comm<LO> > SerialComm = rcp(new MpiComm<LO>(MPI_COMM_SELF));
        RCP<const Comm<LO> > GlobalComm = rowMap->getComm();
        RCP<const Map<LO,GO,NO>> innerMap = MapFactory<LO,GO,NO>::Build(rowMap->lib(),Teuchos::OrdinalTraits<GO>::invalid(), innerNodesArray, baseIndex, GlobalComm);
        RCP<const Map<LO,GO,NO>> overlapMap = MapFactory<LO,GO,NO>::Build(rowMap->lib(),Teuchos::OrdinalTraits<GO>::invalid(), overlapNodesArray, baseIndex, GlobalComm);
        RCP<const Map<LO,GO,NO>> interfaceMap = MapFactory<LO,GO,NO>::Build(rowMap->lib(),Teuchos::OrdinalTraits<GO>::invalid(), interfaceNodesArray, baseIndex, GlobalComm);
        RCP<const Map<LO,GO,NO>> cutMap = MapFactory<LO,GO,NO>::Build(rowMap->lib(),Teuchos::OrdinalTraits<GO>::invalid(), cutNodesArray, baseIndex, GlobalComm);
        RCP<const Map<LO,GO,NO>> matrixImportMap = MapFactory<LO,GO,NO>::Build(rowMap->lib(),Teuchos::OrdinalTraits<GO>::invalid(), matrixImportArray, baseIndex, GlobalComm); 
        // TODO: Undo sorting?
        // sort maps for better debugging
        innerMap = SortMapByGlobalIndex(innerMap);
        overlapMap = SortMapByGlobalIndex(overlapMap);
        interfaceMap = SortMapByGlobalIndex(interfaceMap);
        cutMap = SortMapByGlobalIndex(cutMap);
        matrixImportMap = SortMapByGlobalIndex(matrixImportMap);

        assignMaps(interfaceMap, overlapMap, innerMap, matrixImportMap);

        //TODO: Remove
        bool out_maps=true;
        // #ifndef NDEBUG
        if(out_maps){
            output_map(overlapMap,"ovlpOld");
            output_map(innerMap,"innerOld");
            output_map(interfaceMap,  "interfaceOld");
            output_map(cutMap,  "cutOld");
        }// #endif

        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int HarmonicOverlappingOperator<SC,LO,GO,NO>::calculateHarmonicMapsByRhsHarmonic(){
        FROSCH_DETAILTIMER_START(calculateHarmonicMapsByRhsHarmonicTime,"CalculateHarmonicMapsByRhsHarmonic");
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented!");
        return 0;
    }


    template <class SC,class LO,class GO,class NO>
    RCP<const Map<LO,GO,NO>> HarmonicOverlappingOperator<SC,LO,GO,NO>::calculateInterfacesByRhsHarmonic(RCP<MultiVector<SC,LO,GO,NO>> rhsHarmonic){
        // Prepare
        auto domMap = this->OverlappingMap_;
        const auto & uniqueMap = rhsHarmonic->getMap();

        //Bring the needed rhsHarmonic values on this process
        auto rhsHarmonicExtended = MultiVectorFactory<SC,LO,GO,NO>::Build(domMap,1);
        RCP<Import<LO,GO,NO> >  domImporter = ImportFactory<LO,GO,NO>::Build(uniqueMap,domMap);
        rhsHarmonicExtended->doImport(*rhsHarmonic, *domImporter, Xpetra::CombineMode::INSERT);
        const auto & rhsHarm = rhsHarmonicExtended->getData(0);
        output(rhsHarmonicExtended,"rhsHarmonicExtended",0);

        // Array later containing all interface nodes on this process
        auto interfaceNodesArray = Teuchos::Array<GO>();
        // If the rhs is > eps, the alg. failed to set the rhs to zero there -> need to import residum there.
        // TODO: Problem: We may not do this with a rhs where some parts are zero from the beginning, but can change
        // with another execution. Then we miss nodes and the algorithm doesnt converge
        //TODO: what to choose as epsilon?
        for (LO local=0; local<domMap->getNodeNumElements(); local++){
            if(rhsHarm[local]>1e-10){
                interfaceNodesArray.push_back(domMap->getGlobalElement(local));
            }
        }
        
        //Calculate map
        GO baseIndex = 0;
        RCP<const Comm<LO> > SerialComm = rcp(new MpiComm<LO>(MPI_COMM_SELF));
        RCP<const Comm<LO> > GlobalComm = domMap->getComm();
        RCP<const Map<LO,GO,NO>> interfaceMap = MapFactory<LO,GO,NO>::Build(domMap->lib(),Teuchos::OrdinalTraits<GO>::invalid(), interfaceNodesArray(), baseIndex, GlobalComm);
        interfaceMap = SortMapByGlobalIndex(interfaceMap);

        //TODO: Remove
        bool out_maps=true;
        // #ifndef NDEBUG
        if(out_maps){
            output_map(interfaceMap,  "interfaceMapByRhsHarmonic");
        }// #endif

        return interfaceMap;
    }
}

#endif