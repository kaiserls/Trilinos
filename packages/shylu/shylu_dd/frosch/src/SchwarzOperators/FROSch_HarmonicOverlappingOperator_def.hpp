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
            FROSCH_ASSERT(this->Combine_ != CombinationType::Full, "HarmonicOnOverlap cannot be used with CombinationType==Full")
        }
        std::cout<<"HARMONICCCC"<<std::endl;
    }

    template <class SC,class LO,class GO,class NO>
    int HarmonicOverlappingOperator<SC,LO,GO,NO>::initializeOverlappingOperator(){
        AlgebraicOverlappingOperator<SC,LO,GO,NO>::initializeOverlappingOperator();
        if(HarmonicOnOverlap_){
            if (this->Multiplicity_.is_null()){
                this->calculateMultiplicity();
            }
            // Create needed maps for the nonoverlapping and overlapping part of the domain, cutNodes only needed for restricted paper
            int res = calculateHarmonicMaps<SC,LO,GO,NO>(this->GlobalOverlappingGraph_, this->Multiplicity_, NonOvlpMap_, OvlpMap_, InterfaceMap_, CutNodesMap_);

            // Create importer between the (non)overlapping part and the extendend domain
            OvlpMapper_ = rcp(new Mapper<SC,LO,GO,NO>(this->getDomainMap(),OvlpMap_, this->Multiplicity_));
            NonOvlpMapper_ = rcp(new Mapper<SC,LO,GO,NO>(this->getDomainMap(),NonOvlpMap_, this->Multiplicity_));
        }
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int HarmonicOverlappingOperator<SC,LO,GO,NO>::compute(){
        AlgebraicOverlappingOperator<SC,LO,GO,NO>::compute();
        if(HarmonicOnOverlap_){
            setupHarmonicSolver();
        }
    }

    template <class SC,class LO,class GO,class NO>
    int HarmonicOverlappingOperator<SC,LO,GO,NO>::setupHarmonicSolver(){
        FROSCH_DETAILTIMER_START_LEVELID(setupHarmonicSolver,"OverlappingOperator::setupHarmonicSolver");

        RCP<Matrix<SC,LO,GO,NO>> ovlpMatrix = ExtractLocalSubdomainMatrixNonConst<SC,LO,GO,NO>(this->K_, OvlpMap_);
        // Set dirichlet bc on the interface/cut interface nodes
        ovlpMatrix->resumeFill();
        size_t nEntriesMax = 7;
        Array<LO> localColumns=Array<LO>(nEntriesMax);
        Array<SC> values=Array<SC>(nEntriesMax);
        size_t nEntries;
        for(auto globalRow : InterfaceMap_->getNodeElementList()){
            //retrieve current row
            LO localRow = OvlpMap_->getLocalElement(globalRow);
            ovlpMatrix->getLocalRowCopy(localRow, localColumns(), values(), nEntries);
            //change to dirichlet row
            for(LO i = 0; i<nEntries; i++){
                values[i] = OvlpMap_->getGlobalElement(localColumns[i])==globalRow ? 1.0 : 0.0; //diagonal (dirichlet) : non diagonal
            }
            ovlpMatrix->replaceLocalValues(localRow, localColumns(0,nEntries), values(0,nEntries));
        }
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

            // Import rhs and set dirichlet entries to zero
            RhsPreSolveTmp_->doImport(rhs, *(*OvlpMapper_).Import_, Xpetra::CombineMode::INSERT);//, true
            for(auto globalRow : InterfaceMap_->getNodeElementList()){
                LO localRow = OvlpMap_->getLocalElement(globalRow);
                RhsPreSolveTmp_->replaceLocalValue(localRow, 0, ScalarTraits<SC>::zero());
            }
            OvlpMapper_->setLocalMap(*RhsPreSolveTmp_);
            OvlpMapper_->setLocalMap(*W_);
            HarmonicSolver_->apply(*RhsPreSolveTmp_, *W_, NO_TRANS, ScalarTraits<SC>::one(),ScalarTraits<SC>::zero());
            OvlpMapper_->setGlobalMap(*W_);

            // Bring solution to the full domain:
            Aw->doExport(*W_, *(*OvlpMapper_).Import_, Xpetra::CombineMode::INSERT);//, true, REPLACE?!
            this->K_->apply(*Aw, *Aw);
            rhs.update(-1,*Aw,1);//rhs-A*w

            // output(W_, this->GlobalOverlappingGraph_->getRangeMap(),"w");
            // auto cutVector = MultiVectorFactory<SC,LO,GO,NO>::Build(CutNodesMap_,1);
            // cutVector->putScalar(1.0);
            // output(cutVector, this->GlobalOverlappingGraph_->getRangeMap(),"cut");
            // output(rhsRCP, rhsRCP->getMap(), "unique");
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