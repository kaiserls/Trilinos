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

    template <class SC,class LO,class GO,class NO>
    HarmonicOverlappingOperator<SC,LO,GO,NO>::HarmonicOverlappingOperator(ConstXMatrixPtr k,
                                                          ParameterListPtr parameterList) :
    AlgebraicOverlappingOperator<SC,LO,GO,NO> (k,parameterList)
    {
        HarmonicOnOverlap_ = this->ParameterList_->get("HarmonicOnOverlap",false);
        if(HarmonicOnOverlap_){
            FROSCH_ASSERT(this->Combine_ != CombinationType::Averaging, "HarmonicOnOverlap cannot be used with CombinationType==Averaging")
            //TODO: How to do this better?
            if(this->Combine_ == CombinationType::Restricted){
                this->Combine_ = CombinationType::Full;
                this->rasho = true;
            }
        }
        
    }

    template <class SC,class LO,class GO,class NO>
    int HarmonicOverlappingOperator<SC,LO,GO,NO>::initializeOverlappingOperator(){
        if(HarmonicOnOverlap_){
            if (this->Multiplicity_.is_null()){
                this->calculateMultiplicity();
            }
            // Create needed maps for the nonoverlapping and overlapping part of the domain, cutNodes only needed for restricted paper
            int res = calculateHarmonicMaps<SC,LO,GO,NO>(this->GlobalOverlappingGraph_, this->Multiplicity_, NonOvlpMap_, OvlpMap_, InterfaceMap_, CutNodesMap_, MatrixImportMap_,rasho);
            this->OverlappingMap_=this->MatrixImportMap_;
            //TODO: The line above could introduce a really weired bug if the overlappingMatrix
            // is constructed with the overlappingMap but now we redefine the map
            
            // Create importer between the (non)overlapping part and the extendend domain
            OvlpMapper_ = rcp(new Mapper<SC,LO,GO,NO>(this->getDomainMap(), OvlpMap_, OvlpMap_, OvlpMap_, this->Multiplicity_, CombinationType::Restricted));
            NonOvlpMapper_ = rcp(new Mapper<SC,LO,GO,NO>(this->getDomainMap(), this->OverlappingMap_, NonOvlpMap_, NonOvlpMap_, this->Multiplicity_, this->Combine_));
            //TODO: Eliminate need for these two mappers!
            UniqueToNonOvlpMapper_ = rcp(new Mapper<SC,LO,GO,NO>(this->getDomainMap(), NonOvlpMap_, NonOvlpMap_, NonOvlpMap_, this->Multiplicity_, this->Combine_));
            NonOvlpToOvlpMapper_ = rcp(new Mapper<SC,LO,GO,NO>(NonOvlpMap_, this->OverlappingMap_, this->OverlappingMap_, this->OverlappingMap_, this->Multiplicity_, this->Combine_));

            // TODO: Remove debugging
            #ifndef NDEBUG
            writeMeta(this->GlobalOverlappingGraph_->getRangeMap());
            
            output_map(this->GlobalOverlappingGraph_->getRangeMap(),"unique");
            output_map(this->OverlappingMap_,"overlapping");        
            output_map(OvlpMap_,"ovlp");
            output_map(NonOvlpMap_,"nonOvlp");
            output_map(InterfaceMap_,  "interface");
            output_map(CutNodesMap_,  "cut");
            #endif

            RCP<FancyOStream> wrappedCout = getFancyOStream (rcpFromRef (std::cout)); // Wrap std::cout in a FancyOStream.
            // OvlpMap_->describe(*wrappedCout, Teuchos::VERB_EXTREME);
            // NonOvlpMap_->describe(*wrappedCout, Teuchos::VERB_EXTREME);
            // InterfaceMap_->describe(*wrappedCout, Teuchos::VERB_EXTREME);
            // CutNodesMap_->describe(*wrappedCout, Teuchos::VERB_EXTREME);
            // MatrixImportMap_->describe(*wrappedCout, Teuchos::VERB_EXTREME);
        }
        AlgebraicOverlappingOperator<SC,LO,GO,NO>::initializeOverlappingOperator();
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

        //TODO: Remove debugging
        RCP<FancyOStream> wrappedCout = getFancyOStream (rcpFromRef (std::cout)); // Wrap std::cout in a FancyOStream.
        // MatrixImportMap_->describe(*wrappedCout, Teuchos::VERB_EXTREME);
        // this->K_->describe(*wrappedCout, Teuchos::VERB_DEFAULT);

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
            // TODO: Vector of NonOverlappingMap definieren, import in zwei schritten:
            if (IntermedNonOvlp_.is_null()) {
                IntermedNonOvlp_ = MultiVectorFactory<SC,LO,GO,NO>::Build(NonOvlpMap_,x.getNumVectors());
            } else {
                IntermedNonOvlp_->putScalar(ScalarTraits<SC>::zero());
            }
            // - unique -> nonoverlap
            UniqueToNonOvlpMapper_->restrict(this->XTmp_, IntermedNonOvlp_);
            // nonoverlapp -> overlap
            NonOvlpToOvlpMapper_->restrict(IntermedNonOvlp_,this->XOverlap_);
            #ifndef NDEBUG
            outputWithOtherMap(this->XOverlap_, this->OverlappingMap_, "XOverlap_New", iteration);
            #endif
            // Oder sogar mit forschleife selbstgeschrieben
        } else{
            this->Mapper_->restrict(this->XTmp_, this->XOverlap_); //Restrict into the local overlapping subdomain vector
        }
        #ifndef NDEBUG
        //output fix, because XOverlap is local aver restrict and was never intended for output/export
        outputWithOtherMap(this->XOverlap_, this->OverlappingMap_, "XOverlap_", iteration);
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
        //std::cout<<"           setup harmonic solver"<<std::endl;
        RCP<Matrix<SC,LO,GO,NO>> ovlpMatrix = ExtractLocalSubdomainMatrixNonConst<SC,LO,GO,NO>(this->K_, OvlpMap_);
        //std::cout<<"extract local subdomain map finished"<<std::endl;
        
        
        //ovlpMatrix->getRowMap()->describe(*wrappedCout, Teuchos::VERB_EXTREME);
        // Set dirichlet bc on the interface/cut interface nodes
        ovlpMatrix->resumeFill();
        // Commented out, because i dont import values from the interface nodes now
        // size_t nEntriesMax = 7;
        // Array<LO> localColumns=Array<LO>(nEntriesMax);
        // Array<SC> values=Array<SC>(nEntriesMax);
        // size_t nEntries;
        // for(auto globalRow : InterfaceMap_->getNodeElementList()){
        //     //retrieve current row
        //     LO localRow = OvlpMap_->getLocalElement(globalRow);//ovlpMatrix has the same local "order" as OvlpMap_
        //     std::cout<<"Local row" <<localRow<<std::endl;
        //     ovlpMatrix->getLocalRowCopy(localRow, localColumns(), values(), nEntries);
        //     //change to dirichlet row
        //     for(LO i = 0; i<nEntries; i++){
        //         values[i] = OvlpMap_->getGlobalElement(localColumns[i])==globalRow ? 1.0 : 0.0; //diagonal (dirichlet) : non diagonal
        //     }
        //     //TODO: print out localRow, nEntries, column indices and compare with error message
        //     std::cout<<"local Row "<<localRow<<" local Columns "<<localColumns(0,nEntries)<<"..."<<std::endl;
        //     //TODO: vergleiche ovlpmap_ und maps der matrix
        //     ovlpMatrix->replaceLocalValues(localRow, localColumns(0,nEntries), values(0,nEntries));
        // }
        //std::cout<<"              one mroe step: fill complete"<<std::endl;

        ovlpMatrix->fillComplete();
        HarmonicSolver_ = SolverFactory<SC,LO,GO,NO>::Build(ovlpMatrix,
                                                             sublist(this->ParameterList_,"Solver"),
                                                             string("Solver (Level ") + to_string(this->LevelID_) + string(")"));
        HarmonicSolver_->initialize();
        HarmonicSolver_->compute();
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
            W_ = MultiVectorFactory<SC,LO,GO,NO>::Build(OvlpMap_,1); //Build(OverlappingMatrix_->getDomainMap(),1);
            RhsPreSolveTmp_= MultiVectorFactory<SC,LO,GO,NO>::Build(OvlpMap_, 1);
            RCP<XMultiVector> rhsRCP = RCP<XMultiVector>(&rhs, false);
            auto Aw = MultiVectorFactory<SC,LO,GO,NO>::Build(rhs.getMap(),1);
            #ifndef NDEBUG
            auto rank = rhs.getMap()->getComm()->getRank();
            if(rank==0){
                std::cout<<"ovlpmapper import"<<std::endl;
            }
            #endif
            // Import rhs and set dirichlet entries to zero
            RhsPreSolveTmp_->doImport(rhs, *(*OvlpMapper_).Import_, Xpetra::CombineMode::INSERT);//, true
            //TODO: Do this for restricted?
            // for(auto globalRow : InterfaceMap_->getNodeElementList()){
            //     LO localRow = OvlpMap_->getLocalElement(globalRow);
            //     RhsPreSolveTmp_->replaceLocalValue(localRow, 0, ScalarTraits<SC>::zero());
            // }
            OvlpMapper_->setLocalMap(*RhsPreSolveTmp_);
            OvlpMapper_->setLocalMap(*W_);
            HarmonicSolver_->apply(*RhsPreSolveTmp_, *W_, NO_TRANS, ScalarTraits<SC>::one(),ScalarTraits<SC>::zero());
            OvlpMapper_->setGlobalMap(*W_);

            // Bring solution to the full domain:
            Aw->doExport(*W_, *(*OvlpMapper_).Import_, Xpetra::CombineMode::INSERT);//, true, REPLACE?!
            this->K_->apply(*Aw, *Aw);
            #ifndef NDEBUG
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
            wUnique->doExport(*W_, *(*OvlpMapper_).Import_, Xpetra::CombineMode::INSERT);
            lhs.update(1.,*wUnique, 1.);//lhs+W_
        }
    }

    template <class SC,class LO,class GO,class NO>
    string HarmonicOverlappingOperator<SC,LO,GO,NO>::description() const
    {
        return "Harmonic Overlap Operator";
    }
}

#endif