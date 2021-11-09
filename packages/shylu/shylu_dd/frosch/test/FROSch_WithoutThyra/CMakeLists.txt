TRIBITS_ADD_EXECUTABLE(
frosch_withoutThyra
SOURCES main.cpp
)

TRIBITS_ADD_TEST(
frosch_withoutThyra
NAME test_frosch_withoutThyra
ARGS "--M=5 --DIM=2 --O=2 --DPN=1 --PLIST=ParameterList_OneLevelPreconditioner_Matrix.xml --USETPETRA"
COMM mpi
NUM_MPI_PROCS 4
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(WithoutThyraCopyFiles
DEST_FILES ParameterList_OneLevelPreconditioner_Matrix.xml
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
    EXEDEPS frosch_withoutThyra
)
