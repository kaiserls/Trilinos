TRIBITS_COPY_FILES_TO_BINARY_DIR(OverlapCopyFiles
DEST_FILES ParameterList_OneLevelPreconditioner_Graph.xml
           ParameterList_OneLevelPreconditioner_Matrix.xml
           ParameterList_OneLevelPreconditioner_Old.xml
           ParameterList_OneLevelPreconditioner_cg.xml
           ParameterList_OneLevelPreconditioner_asho.xml
           ParameterList_OneLevelPreconditioner_rasho.xml
           ParameterList_OneLevelPreconditioner_ras.xml
           ParameterList_OneLevelPreconditioner_cg_NonRepeated.xml
           ParameterList_OneLevelPreconditioner_asho_byMultiplicity.xml
           ParameterList_OneLevelPreconditioner_asho_byRhsHarmonic.xml
           ParameterList_OneLevelPreconditioner_asho_onOverlapping.xml
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
    EXEDEPS overlap
)
