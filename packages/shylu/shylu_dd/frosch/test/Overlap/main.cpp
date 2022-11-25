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

#include <ShyLU_DDFROSch_config.h>

#include <mpi.h>

#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_XMLParameterListCoreHelpers.hpp>
#include <Teuchos_StackedTimer.hpp>

// Galeri::Xpetra
#include "Galeri_XpetraProblemFactory.hpp"
#include "Galeri_XpetraMatrixTypes.hpp"
#include "Galeri_XpetraParameters.hpp"
#include "Galeri_XpetraUtils.hpp"
#include "Galeri_XpetraMaps.hpp"

// Thyra includes
#include <Thyra_LinearOpWithSolveBase.hpp>
#include <Thyra_VectorBase.hpp>
#include <Thyra_SolveSupportTypes.hpp>
#include <Thyra_LinearOpWithSolveBase.hpp>
#include <Thyra_LinearOpWithSolveFactoryHelpers.hpp>
#include <Thyra_TpetraLinearOp.hpp>
#include <Thyra_TpetraMultiVector.hpp>
#include <Thyra_TpetraVector.hpp>
#include <Thyra_TpetraThyraWrappers.hpp>
#include <Thyra_VectorBase.hpp>
#include <Thyra_VectorStdOps.hpp>
#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
#include <Thyra_EpetraLinearOp.hpp>
#endif
#include <Thyra_VectorSpaceBase_def.hpp>
#include <Thyra_VectorSpaceBase_decl.hpp>

// Stratimikos includes
#include <Stratimikos_FROSch_def.hpp>

#include <Tpetra_Core.hpp>

// Xpetra include
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_DefaultPlatform.hpp>
#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
#include <Xpetra_EpetraCrsMatrix.hpp>
#endif
#include <Xpetra_Parameters.hpp>

// FROSCH thyra includes
#include "Thyra_FROSchLinearOp_def.hpp"
#include "Thyra_FROSchFactory_def.hpp"
#include <FROSch_Tools_def.hpp>

#include "FROSch_Debugging.hpp"

using UN    = unsigned;
using SC    = double;
using LO    = int;
using GO    = FROSch::DefaultGlobalOrdinal;
using NO    = KokkosClassic::DefaultNode::DefaultNodeType;

using namespace std;
using namespace Teuchos;
using namespace Xpetra;
using namespace FROSch;
using namespace Thyra;

SC exact_sol(SC x, SC y)
{
    return exp(5*(x+y))*sin(M_PI*x)*sin(M_PI*y);
}

SC rhs(SC x, SC y)
{
    return 2.*exp(5.* (x + y))*(-5.*M_PI*cos(M_PI*y) * sin(M_PI*x) + (-5.*M_PI*cos(M_PI*x) + (-25. + M_PI*M_PI)*sin(M_PI*x))*sin(M_PI*y));
}

int main(int argc, char *argv[])
{
    oblackholestream blackhole;
    GlobalMPISession mpiSession(&argc,&argv,&blackhole);

    RCP<const Comm<int> > CommWorld = DefaultPlatform::getDefaultPlatform().getComm();

    CommandLineProcessor My_CLP;

    RCP<FancyOStream> out = VerboseObjectBase::getDefaultOStream();

    int M = 3;
    My_CLP.setOption("M",&M,"H / h.");
    int Dimension = 2;
    My_CLP.setOption("DIM",&Dimension,"Dimension.");
    int Overlap = 0;
    My_CLP.setOption("O",&Overlap,"Overlap.");
    int NumberOfBlocks = 1;
    My_CLP.setOption("NB",&NumberOfBlocks,"Number of blocks.");
    int DofsPerNode = 1;
    My_CLP.setOption("DPN",&DofsPerNode,"Dofs per node.");
    int DOFOrdering = 0;
    My_CLP.setOption("ORD",&DOFOrdering,"Dofs ordering (NodeWise=0, DimensionWise=1, Custom=2).");
    string xmlFile = "ParameterList.xml";
    My_CLP.setOption("PLIST",&xmlFile,"File name of the parameter list.");
    bool useepetra = false;
    My_CLP.setOption("USEEPETRA","USETPETRA",&useepetra,"Use Epetra infrastructure for the linear algebra.");

    My_CLP.recogniseAllOptions(true);
    My_CLP.throwExceptions(false);
    CommandLineProcessor::EParseCommandLineReturn parseReturn = My_CLP.parse(argc,argv);
    if (parseReturn == CommandLineProcessor::PARSE_HELP_PRINTED) {
        return(EXIT_SUCCESS);
    }

    CommWorld->barrier();
    RCP<StackedTimer> stackedTimer = rcp(new StackedTimer("Overlap Test"));
    TimeMonitor::setStackedTimer(stackedTimer);

    int N = 0;
    int color=1;
    int mx=0;
    int my=0;
    if (Dimension == 2) {
        N = (int) (pow(CommWorld->getSize(),1/2.) + 100*numeric_limits<double>::epsilon()); // 1/H
        if (CommWorld->getRank()<N*N) {
            color=0;
            mx=N;
            my=N;
        }
        if(CommWorld->getSize()==2){
            color=0;
            mx=2;
            my=1;
        }
    } else if (Dimension == 3) {
        N = (int) (pow(CommWorld->getSize(),1/3.) + 100*numeric_limits<double>::epsilon()); // 1/H
        if (CommWorld->getRank()<N*N*N) {
            color=0;
        }
    } else {
        assert(false);
    }

    UnderlyingLib xpetraLib = UseTpetra;
    if (useepetra) {
        xpetraLib = UseEpetra;
    } else {
        xpetraLib = UseTpetra;
    }

    RCP<const Comm<int> > Comm = CommWorld->split(color,CommWorld->getRank());

    if (color==0) {

        RCP<ParameterList> parameterList = getParametersFromXmlFile(xmlFile);

        ArrayRCP<RCP<Matrix<SC,LO,GO,NO> > > K(NumberOfBlocks);
        ArrayRCP<RCP<Map<LO,GO,NO> > > RepeatedMaps(NumberOfBlocks);
        ArrayRCP<RCP<Map<LO,GO,NO> > > UniqueMaps(NumberOfBlocks);
        ArrayRCP<RCP<MultiVector<SC,LO,GO,NO> > > Coordinates(NumberOfBlocks);
        ArrayRCP<UN> dofsPerNodeVector(NumberOfBlocks);

        const GO INVALID = Teuchos::OrdinalTraits<GO>::invalid();
        for (UN block=0; block<(UN) NumberOfBlocks; block++) {
            Comm->barrier(); if (Comm->getRank()==0) cout << "###################\n# Assembly Block " << block << " #\n###################\n" << endl;

            dofsPerNodeVector[block] = (UN) max(int(DofsPerNode-block),1);

            ParameterList GaleriList;
            GaleriList.set("nx", GO(mx*(M+block)));
            GaleriList.set("ny", GO(my*(M+block)));
            GaleriList.set("nz", GO(N*(M+block)));
            GaleriList.set("mx", GO(mx));
            GaleriList.set("my", GO(my));
            GaleriList.set("mz", GO(N));

            RCP<const Map<LO,GO,NO> > UniqueMapTmp;
            RCP<MultiVector<SC,LO,GO,NO> > CoordinatesTmp;
            RCP<Matrix<SC,LO,GO,NO> > KTmp;
            if (Dimension==2) {
                UniqueMapTmp = Galeri::Xpetra::CreateMap<LO,GO,NO>(xpetraLib,"Cartesian2D",Comm,GaleriList); // RCP<FancyOStream> fancy = fancyOStream(rcpFromRef(cout)); nodeMap->describe(*fancy,VERB_EXTREME);
                CoordinatesTmp = Galeri::Xpetra::Utils::CreateCartesianCoordinates<SC,LO,GO,Map<LO,GO,NO>,MultiVector<SC,LO,GO,NO> >("2D",UniqueMapTmp,GaleriList);
                RCP<Galeri::Xpetra::Problem<Map<LO,GO,NO>,CrsMatrixWrap<SC,LO,GO,NO>,MultiVector<SC,LO,GO,NO> > > Problem = Galeri::Xpetra::BuildProblem<SC,LO,GO,Map<LO,GO,NO>,CrsMatrixWrap<SC,LO,GO,NO>,MultiVector<SC,LO,GO,NO> >("Laplace2D",UniqueMapTmp,GaleriList);
                KTmp = Problem->BuildMatrix();
            } else if (Dimension==3) {
                UniqueMapTmp = Galeri::Xpetra::CreateMap<LO,GO,NO>(xpetraLib,"Cartesian3D",Comm,GaleriList); // RCP<FancyOStream> fancy = fancyOStream(rcpFromRef(cout)); nodeMap->describe(*fancy,VERB_EXTREME);
                CoordinatesTmp = Galeri::Xpetra::Utils::CreateCartesianCoordinates<SC,LO,GO,Map<LO,GO,NO>,MultiVector<SC,LO,GO,NO> >("3D",UniqueMapTmp,GaleriList);
                RCP<Galeri::Xpetra::Problem<Map<LO,GO,NO>,CrsMatrixWrap<SC,LO,GO,NO>,MultiVector<SC,LO,GO,NO> > > Problem = Galeri::Xpetra::BuildProblem<SC,LO,GO,Map<LO,GO,NO>,CrsMatrixWrap<SC,LO,GO,NO>,MultiVector<SC,LO,GO,NO> >("Laplace3D",UniqueMapTmp,GaleriList);
                KTmp = Problem->BuildMatrix();
                // RCP<FancyOStream> fancy = fancyOStream(rcpFromRef(cout));
                // KTmp->describe(*fancy,VERB_EXTREME);
            }

            RCP<Map<LO,GO,NO> > UniqueMap;

            if (DOFOrdering == 0) {
                Array<GO> uniqueMapArray(dofsPerNodeVector[block]*UniqueMapTmp->getLocalNumElements());
                for (LO i=0; i<(LO) UniqueMapTmp->getLocalNumElements(); i++) {
                    for (UN j=0; j<dofsPerNodeVector[block]; j++) {
                        uniqueMapArray[dofsPerNodeVector[block]*i+j] = dofsPerNodeVector[block]*UniqueMapTmp->getGlobalElement(i)+j;
                    }
                }

                UniqueMap = MapFactory<LO,GO,NO>::Build(xpetraLib,INVALID,uniqueMapArray(),0,Comm);
                K[block] = MatrixFactory<SC,LO,GO,NO>::Build(UniqueMap,KTmp->getGlobalMaxNumRowEntries());
                for (LO i=0; i<(LO) UniqueMapTmp->getLocalNumElements(); i++) {
                    ArrayView<const LO> indices;
                    ArrayView<const SC> values;
                    KTmp->getLocalRowView(i,indices,values);

                    for (UN j=0; j<dofsPerNodeVector[block]; j++) {
                        Array<GO> indicesArray(indices.size());
                        for (LO k=0; k<indices.size(); k++) {
                            indicesArray[k] = dofsPerNodeVector[block]*KTmp->getColMap()->getGlobalElement(indices[k])+j;
                        }
                        K[block]->insertGlobalValues(dofsPerNodeVector[block]*KTmp->getRowMap()->getGlobalElement(i)+j,indicesArray(),values);
                    }
                }
                K[block]->fillComplete();
            } else if (DOFOrdering == 1) {
                Array<GO> uniqueMapArray(dofsPerNodeVector[block]*UniqueMapTmp->getLocalNumElements());
                for (LO i=0; i<(LO) UniqueMapTmp->getLocalNumElements(); i++) {
                    for (UN j=0; j<dofsPerNodeVector[block]; j++) {
                        uniqueMapArray[i+UniqueMapTmp->getLocalNumElements()*j] = UniqueMapTmp->getGlobalElement(i)+(UniqueMapTmp->getMaxAllGlobalIndex()+1)*j;
                    }
                }

                UniqueMap = MapFactory<LO,GO,NO>::Build(xpetraLib,INVALID,uniqueMapArray(),0,Comm);
                K[block] = MatrixFactory<SC,LO,GO,NO>::Build(UniqueMap,KTmp->getGlobalMaxNumRowEntries());
                for (LO i=0; i<(LO) UniqueMapTmp->getLocalNumElements(); i++) {
                    ArrayView<const LO> indices;
                    ArrayView<const SC> values;
                    KTmp->getLocalRowView(i,indices,values);

                    for (UN j=0; j<dofsPerNodeVector[block]; j++) {
                        Array<GO> indicesArray(indices.size());
                        for (LO k=0; k<indices.size(); k++) {
                            indicesArray[k] = KTmp->getColMap()->getGlobalElement(indices[k])+(KTmp->getColMap()->getMaxAllGlobalIndex()+1)*j;
                        }
                        K[block]->insertGlobalValues(UniqueMapTmp->getGlobalElement(i)+(UniqueMapTmp->getMaxAllGlobalIndex()+1)*j,indicesArray(),values);
                    }
                }
                K[block]->fillComplete();
            } else if (DOFOrdering == 2) {
                assert(false); // TODO: Andere Sortierung implementieren
            } else {
                assert(false);
            }
            Coordinates[block] = CoordinatesTmp;
            UniqueMaps[block]   = UniqueMap;
            RepeatedMaps[block] = BuildRepeatedMapNonConst<LO,GO,NO>(K[block]->getCrsGraph()); //RCP<FancyOStream> fancy = fancyOStream(rcpFromRef(cout)); RepeatedMaps[block]->describe(*fancy,VERB_EXTREME);
        }

        Comm->barrier(); if (Comm->getRank()==0) cout << "##############################\n# Assembly Monolithic System #\n##############################\n" << endl;

        RCP<Matrix<SC,LO,GO,NO> > KMonolithic;
        if (NumberOfBlocks>1) {

            Array<GO> uniqueMapArray(0);
            GO tmpOffset = 0;
            for (UN block=0; block<(UN) NumberOfBlocks; block++) {
                ArrayView<const GO> tmpgetGlobalElements = K[block]->getMap()->getLocalElementList();
                for (LO i=0; i<tmpgetGlobalElements.size(); i++) {
                    uniqueMapArray.push_back(tmpgetGlobalElements[i]+tmpOffset);
                }
                tmpOffset += K[block]->getMap()->getMaxAllGlobalIndex()+1;
            }
            RCP<Map<LO,GO,NO> > UniqueMapMonolithic = MapFactory<LO,GO,NO>::Build(xpetraLib,INVALID,uniqueMapArray(),0,Comm);

            tmpOffset = 0;
            KMonolithic = MatrixFactory<SC,LO,GO,NO>::Build(UniqueMapMonolithic,K[0]->getGlobalMaxNumRowEntries());
            for (UN block=0; block<(UN) NumberOfBlocks; block++) {
                for (LO i=0; i<(LO) K[block]->getLocalNumRows(); i++) {
                    ArrayView<const LO> indices;
                    ArrayView<const SC> values;
                    K[block]->getLocalRowView(i,indices,values);
                    Array<GO> indicesGlobal(indices.size());
                    for (UN j=0; j<indices.size(); j++) {
                        indicesGlobal[j] = K[block]->getColMap()->getGlobalElement(indices[j])+tmpOffset;
                    }
                    KMonolithic->insertGlobalValues(K[block]->getMap()->getGlobalElement(i)+tmpOffset,indicesGlobal(),values);
                }
                tmpOffset += K[block]->getMap()->getMaxAllGlobalIndex()+1;
            }
            KMonolithic->fillComplete();
        } else if (NumberOfBlocks==1) {
            KMonolithic = K[0];
        } else {
            assert(false);
        }

        // RCP<FancyOStream> fancy = fancyOStream(rcpFromRef(cout));
        // KMonolithic->describe(*fancy,VERB_EXTREME);
        RCP<MultiVector<SC,LO,GO,NO> > xExact = MultiVectorFactory<SC,LO,GO,NO>::Build(KMonolithic->getMap(),1);
        RCP<MultiVector<SC,LO,GO,NO> > xSolution = MultiVectorFactory<SC,LO,GO,NO>::Build(KMonolithic->getMap(),1);
        RCP<MultiVector<SC,LO,GO,NO> > xRightHandSide = MultiVectorFactory<SC,LO,GO,NO>::Build(KMonolithic->getMap(),1);
        RCP<MultiVector<SC,LO,GO,NO> > xSolutionModified = MultiVectorFactory<SC,LO,GO,NO>::Build(KMonolithic->getMap(),1);
        RCP<MultiVector<SC,LO,GO,NO> > xRightHandSideModified = MultiVectorFactory<SC,LO,GO,NO>::Build(KMonolithic->getMap(),1);

        xSolution->putScalar(ScalarTraits<SC>::zero());
        xRightHandSide->putScalar(ScalarTraits<SC>::one());
        xSolutionModified->putScalar(ScalarTraits<SC>::zero());
        xRightHandSideModified->putScalar(ScalarTraits<SC>::one());

        // TODO: Push code?
        auto xVec = Coordinates[0]->getData(0);
        auto yVec = Coordinates[0]->getData(1);

        auto h2 = (xVec[1] - xVec[0])*(xVec[1] - xVec[0]);

        auto f1 = xRightHandSide->getDataNonConst(0);
        auto f2 = xRightHandSideModified->getDataNonConst(0);
        auto ex = xExact->getDataNonConst(0);

        for(UN i=0; i<xVec.size(); i++){
            SC f = rhs(xVec[i], yVec[i]);
            f1[i]=f;
            f2[i]=f;

            SC exact = exact_sol(xVec[i],yVec[i]);
            ex[i] = exact;
        }

        xRightHandSide->scale(h2);
        xRightHandSideModified->scale(h2);

        // KMonolithic->scale(1./h2);

        CrsMatrixWrap<SC,LO,GO,NO>& crsWrapK = dynamic_cast<CrsMatrixWrap<SC,LO,GO,NO>&>(*KMonolithic);
        RCP<const LinearOpBase<SC> > K_thyra = ThyraUtils<SC,LO,GO,NO>::toThyra(crsWrapK.getCrsMatrix());
        RCP<MultiVectorBase<SC> >thyraX = rcp_const_cast<MultiVectorBase<SC> >(ThyraUtils<SC,LO,GO,NO>::toThyraMultiVector(xSolution));
        RCP<MultiVectorBase<SC> >thyraB = rcp_const_cast<MultiVectorBase<SC> >(ThyraUtils<SC,LO,GO,NO>::toThyraMultiVector(xRightHandSide));

        RCP<MultiVectorBase<SC> >thyraXModified = rcp_const_cast<MultiVectorBase<SC> >(ThyraUtils<SC,LO,GO,NO>::toThyraMultiVector(xSolutionModified));
        RCP<MultiVectorBase<SC> >thyraBModified = rcp_const_cast<MultiVectorBase<SC> >(ThyraUtils<SC,LO,GO,NO>::toThyraMultiVector(xRightHandSideModified));
        
        RCP<MultiVectorBase<SC> >thyraXExact = rcp_const_cast<MultiVectorBase<SC> >(ThyraUtils<SC,LO,GO,NO>::toThyraMultiVector(xExact));
        //-----------Set Coordinates and RepMap in ParameterList--------------------------
        RCP<ParameterList> plList =  sublist(parameterList,"Preconditioner Types");
        sublist(plList,"FROSch")->set("Dimension",Dimension);
        sublist(plList,"FROSch")->set("Overlap",Overlap);
        if (NumberOfBlocks>1) {
            sublist(plList,"FROSch")->set("Repeated Map Vector",RepeatedMaps);

            ArrayRCP<DofOrdering> dofOrderings(NumberOfBlocks);
            if (DOFOrdering == 0) {
                for (UN block=0; block<(UN) NumberOfBlocks; block++) {
                    dofOrderings[block] = NodeWise;
                }
            } else if (DOFOrdering == 1) {
                for (UN block=0; block<(UN) NumberOfBlocks; block++) {
                    dofOrderings[block] = DimensionWise;
                }
            } else {
                assert(false);
            }

            sublist(plList,"FROSch")->set("DofOrdering Vector",dofOrderings);
            sublist(plList,"FROSch")->set("DofsPerNode Vector",dofsPerNodeVector);
        } else if (NumberOfBlocks==1) {
            sublist(plList,"FROSch")->set("Repeated Map",RepeatedMaps[0]);
            // sublist(plList,"FROSch")->set("Coordinates List",Coordinates[0]); // Does not work yet...

            string DofOrderingString;
            if (DOFOrdering == 0) {
                DofOrderingString = "NodeWise";
            } else if (DOFOrdering == 1) {
                DofOrderingString = "DimensionWise";
            } else {
                assert(false);
            }
            sublist(plList,"FROSch")->set("DofOrdering",DofOrderingString);
            sublist(plList,"FROSch")->set("DofsPerNode",DofsPerNode);
        } else {
            assert(false);
        }

        Comm->barrier();
        if (Comm->getRank()==0) {
            cout << "##################\n# Parameter List #\n##################" << endl;
            parameterList->print(cout);
            cout << endl;
        }

        double start = MPI_Wtime();

        Comm->barrier(); if (Comm->getRank()==0) cout << "###################################\n# Stratimikos LinearSolverBuilder #\n###################################\n" << endl;
        Stratimikos::LinearSolverBuilder<SC> linearSolverBuilder;
        Stratimikos::enableFROSch<SC,LO,GO,NO>(linearSolverBuilder);
        linearSolverBuilder.setParameterList(parameterList);
        // Comm->barrier(); if (Comm->getRank()==0) cout << "######################\n# Thyra PrepForSolve #\n######################\n" << endl
        RCP<LinearOpWithSolveFactoryBase<SC> > solverFactory =
        Thyra::createLinearSolveStrategy(linearSolverBuilder);

        solverFactory->setOStream(out);
        solverFactory->setVerbLevel(VERB_HIGH);
        // Comm->barrier(); if (Comm->getRank()==0) cout << "###########################\n# Thyra LinearOpWithSolve #\n###########################" << endl
        // preconditioner
        auto precFactory = solverFactory->getPreconditionerFactory();
        RCP<Thyra::PreconditionerBase<SC> > prec = precFactory->createPrec();
        Thyra::initializePrec<SC>(*precFactory, K_thyra, prec.ptr());
        // solver
        // Teuchos::RCP<Thyra::LinearOpWithSolveBase<SC> > solveOp = solverFactory->createOp(); 
        // Thyra::initializePreconditionedOp<double>(*solverFactory, K_thyra, prec, solveOp.ptr());
        // Comm->barrier(); if (Comm->getRank()==0) cout << "###########################\n# Casting #\n###########################" << endl;
        
        //op to FROSch Operator
        auto nonConstOp = rcp_const_cast<LinearOpBase<SC>>(prec->getUnspecifiedPrecOp());
        auto froschLinearOp = rcp_dynamic_cast<Thyra::FROSchLinearOp<SC, LO, GO, NO>,LinearOpBase<SC>>(nonConstOp, true);
        
        // Comm->barrier(); if (Comm->getRank()==0) cout << "###########################\n# PreSolve #\n###########################" << endl;
        SC residualBeforePre=norm(*(thyraBModified->col(0)));
        // Preconditioner is called from FROSch Operator
        froschLinearOp->preSolve(thyraBModified.ptr());
        SC residualAfterPre=norm(*(thyraBModified->col(0)));
        
        if (Comm->getRank()==0) cout<<"Norm of rhs before: " << residualBeforePre << "; Norm of rhs after preSolve: " <<residualAfterPre << endl;
        
        double adapt_start = MPI_Wtime();
        // Adapt convergence tolerance
        string solverType = sublist(sublist(parameterList,"Linear Solver Types"),"Belos")->get<string>("Solver Type");
        RCP<ParameterList> krylovList =  sublist(sublist(sublist(sublist(parameterList,"Linear Solver Types"),"Belos"),"Solver Types"),solverType, true);
        double convergenceTolerance = krylovList->get<double>("Convergence Tolerance");
        double scaledConvergenceTolerance = convergenceTolerance *  residualBeforePre / residualAfterPre;
        krylovList->set("Convergence Tolerance", scaledConvergenceTolerance);
        // Re setup solver
        linearSolverBuilder.setParameterList(parameterList);
        solverFactory = Thyra::createLinearSolveStrategy(linearSolverBuilder);
        Teuchos::RCP<Thyra::LinearOpWithSolveBase<SC> > solveOp = solverFactory->createOp(); 
        Thyra::initializePreconditionedOp<double>(*solverFactory, K_thyra, prec, solveOp.ptr());
        double adapt_end = MPI_Wtime();
        if (Comm->getRank()==0) cout << "The code block for thes caling and setup of solveOp took "<< adapt_end-adapt_start<<"seconds"<<endl;

        // Comm->barrier(); if (Comm->getRank()==0) cout << "\n#########\n# Solve #\n#########" << endl;
        SolveStatus<SC> status =
        solve<SC>(*solveOp, Thyra::NOTRANS, *thyraBModified, thyraXModified.ptr());
        FROSCH_ASSERT(status.solveStatus==SOLVE_STATUS_CONVERGED, "Solver didn't converge");
        //Comm->barrier(); if (Comm->getRank()==0) cout << "the error is: - for the new system:" << thyraB->col(0)->norm_1() << " for the olds system " << thyraB->col(0)->norm_1()<<std::endl;

        // Comm->barrier(); if (Comm->getRank()==0) cout << "###########################\n# AfterSolve #\n###########################" << endl;
        thyraX = thyraXModified->clone_mv();
        froschLinearOp->afterSolve(thyraX.ptr());

        double end = MPI_Wtime();
        if (Comm->getRank()==0) cout << "Solving the system took " << end - start << " seconds to run." << endl;

        // check if it solves the original problem:
        K_thyra->apply(EOpTransp::NOTRANS,*thyraX, thyraB.ptr(),1.0,-1.0);
        K_thyra->apply(EOpTransp::NOTRANS,*thyraXModified, thyraBModified.ptr(),1.0,-1.0);
        SC residualModifiedSystem=norm(*(thyraBModified->col(0)));
        SC residualOriginalSystem=norm(*(thyraB->col(0)));


        SC errorBefore = norm(*(thyraXExact->col(0)));
        update(-1., *thyraX, thyraXExact.ptr());
        //auto xpetravec = FROSch::toXpetra<SC,LO,GO,NO>(thyraXExact.ptr()); // Teuchos::RCP<Thyra::MultiVectorBase<SC>>
        //output(xpetravec, "xExact", 0);

        SC errorAfter = norm(*(thyraXExact->col(0)));

        Comm->barrier(); if (Comm->getRank()==0) cout << "the absolut residual is: - for the Modified system: "<<residualModifiedSystem << " - for the original system: "<< residualOriginalSystem <<std::endl;
        Comm->barrier(); if (Comm->getRank()==0) cout << "the error to the exact solution is: "<< errorAfter << ". Reduced from: " << errorBefore << std::endl;
        Comm->barrier(); if (Comm->getRank()==0) cout << "\n#############\n# Finished! #\n#############" << endl;
    }
    


    CommWorld->barrier();
    stackedTimer->stop("Overlap Test");
    StackedTimer::OutputOptions options;

    options.output_fraction = true;
    options.output_histogram = options.output_minmax = false;

    stackedTimer->report(*out,CommWorld,options);

    return(EXIT_SUCCESS);

}
