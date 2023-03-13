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

#ifndef _FROSCH_MAPPER_DECL_HPP
#define _FROSCH_MAPPER_DECL_HPP

#include <Xpetra_MapFactory_fwd.hpp>

#include <FROSch_Tools_def.hpp>

namespace FROSch {

    using namespace Teuchos;
    using namespace Xpetra;

    //! Mapper storing maps to execute restriction and prolongation
    template <class SC = double,
              class LO = int,
              class GO = DefaultGlobalOrdinal,
              class NO = KokkosClassic::DefaultNode::DefaultNodeType>
    class Mapper {

    protected:

        using CommPtr               = RCP<const Comm<int> >;
        using XMap                  = Map<LO,GO,NO>;
        using ConstXMapPtr          = RCP<const XMap>;
        using XMultiVector          = MultiVector<SC,LO,GO,NO>;
        using XMultiVectorPtr       = RCP<XMultiVector>;
        using XImportPtr            = RCP<Import<LO,GO,NO>>;
        using XExportPtr            = RCP<Export<LO,GO,NO>>;

        using SCVecPtr              = ArrayRCP<SC>;
        using ConstSCVecPtr         = ArrayRCP<const SC>;
        using UN                    = unsigned;

    public:
        enum OverlapCombinationType {Averaging,Full,Restricted};

        Mapper(ConstXMapPtr uniqueMap, ConstXMapPtr overlappingMap, ConstXMapPtr importMap, ConstXMapPtr exportMap, XMultiVectorPtr multiplicity, OverlapCombinationType combine);
        Mapper(ConstXMapPtr uniqueMap, ConstXMapPtr overlappingMap, XMultiVectorPtr multiplicity, OverlapCombinationType combine);

        virtual ~Mapper();

        virtual int restrict(const XMultiVectorPtr source, XMultiVectorPtr & target);
        virtual int prolongate(const XMultiVectorPtr source, XMultiVectorPtr & target);
        virtual int insertInto(const XMultiVectorPtr uniqueSource, XMultiVectorPtr & target);
        virtual int insertIntoWithCheck(const XMultiVectorPtr uniqueSource, XMultiVectorPtr & target); //, ConstXMapPtr transferMap);

        void setLocalMap(XMultiVector &globalVector);
        void setGlobalMap(XMultiVector &localVector);
    
    //protected: TODO: Make protected, when restrict and prolongate is used!
        // Maps of the source and target vectors
        ConstXMapPtr UniqueMap_; 
        ConstXMapPtr OverlappingMap_;
        ConstXMapPtr OverlappingMapLocal_;

        // Subsets of the above maps used for import/export
        ConstXMapPtr ImportMap_; //Map containing the nodes for which values should be imported
        ConstXMapPtr ExportMap_; //Map containing the nodes for which values should be exported

        // Precalculated imported/export operations
        XImportPtr Import_;
        XExportPtr Export_;//TODO: usefull for rasho?

        XMultiVectorPtr Multiplicity_;//TODO: Move to averagesubclass
        OverlapCombinationType Combine_;

        mutable XMultiVectorPtr XOverlapTmp_;
    };
}

#endif