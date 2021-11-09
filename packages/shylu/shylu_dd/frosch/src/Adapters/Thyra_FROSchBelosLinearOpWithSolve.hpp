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

#ifndef THYRA_FROSCHBELOS_LINEAR_OP_WITH_SOLVE_HPP
#define THYRA_FROSCHBELOS_LINEAR_OP_WITH_SOLVE_HPP

#include "Thyra_BelosLinearOpWithSolve_decl.hpp"
#include "Thyra_BelosLinearOpWithSolve_def.hpp"
#include "FROSch_OneLevelPreconditioner_decl.hpp"

#include "FROSch_Tools_decl.hpp"

namespace Thyra {

template<class Scalar>
class FROSchBelosLinearOpWithSolve : virtual public Thyra::BelosLinearOpWithSolve<Scalar>
{
public:
    SolveStatus<Scalar> solveImpl(
        const EOpTransp M_trans,
        const MultiVectorBase<Scalar> &B,
        const Ptr<MultiVectorBase<Scalar> > &X,
        const Ptr<const SolveCriteria<Scalar> > solveCriteria
    ) const;

};//class froschbelos...


//auto & new_lhs = (*problem.getCurrLHSVec()).copy();

template<class Scalar>
SolveStatus<Scalar>
FROSchBelosLinearOpWithSolve<Scalar>::solveImpl(
  const EOpTransp M_trans,
  const MultiVectorBase<Scalar> &B,
  const Ptr<MultiVectorBase<Scalar> > &X,
  const Ptr<const SolveCriteria<Scalar> > solveCriteria
  ) const {
  std::cout<<"Solving with my custom solver"<<std::endl;
  auto & problem = this->lp_.getProblem();
  auto & prepareP = dynamic_cast<FROSch::OneLevelPreconditioner<SC,LO,GO,NO>>(this->lp_->getLeftPrec());//TODO: Hardcoded if left or right
  
  // prepare right hand side
  RCP<MultiVector<SC,LO,GO,NO>> BXpetraView = FROSch::toXpetra<SC,LO,GO,NO>(&B);
  RCP<MultiVector<SC,LO,GO,NO>> BNew = MultiVectorFactory<SC,LO,GO,NO>::Build(BXpetraView, DataAccess::Copy);
  prepareP.preSolve(BNew);
  
  // solve 
  SolveStatus<Scalar> belosSolveStatus = Thyra::BelosLinearOpWithSolve<Scalar>::solveImpl(M_trans, BNew, X, solveCriteria);
  
  // after solve
  RCP<MultiVector<SC,LO,GO,NO> > XXpetraView =FROSch::toXpetra<SC,LO,GO,NO>(&X);
  prepareP->afterSolve(XXpetraView);  
  
  return belosSolveStatus;
  }

}//namespace thyra

#endif