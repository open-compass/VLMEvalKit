#!/bin/sh

# Get relative path of the root directory of the project
rdir=`git rev-parse --git-dir`
rel_path="$(dirname "$rdir")"
# Change to that path and run the file
cd $rel_path/src/latex2sympy2_extended

# Create gen directory if it doesn't exist
mkdir -p gen

# Run ANTLR on the grammar file
java -jar ../../antlr-4.13.2-complete.jar PS.g4 -o gen/antlr4_13_2
java -jar ../../antlr-4.11.0-complete.jar PS.g4 -o gen/antlr4_11_0
java -jar ../../antlr-4.9.3-complete.jar PS.g4 -o gen/antlr4_9_3