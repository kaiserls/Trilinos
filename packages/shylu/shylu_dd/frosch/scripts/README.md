# Scripts

## Run.py

The run script contains an example, how the overlap testcases can be run for a set of parameters. This allows to compare the performance of the different overlap variants. Adapt the paths and commands to your system and needs before running this script.

## Visualize.py

The visualize script can visualize the files which are written out from the testcases when running ```run.py```. The visualization can be used for debugging and understanding the overlap variants. To generate the output files adapt the line ```bool OutputMapsAndVectors_ = false;``` in the file `FROSch_HarmonicOverlappingOperator_decl.hpp` and recompile your testcase.
