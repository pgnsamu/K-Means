#!/bin/bash

make datasetgen

outputDir="test_files/${1}"
mkdir -p $outputDir

read -p "num elements x row = " num_prt_rows
read -p "num rows = " num_rows
read -p "min val = " min_val
read -p "max val = " max_val

condor_submit job/datasetgen.job  -append "arguments = ${num_prt_rows} ${num_rows} ${min_val} ${max_val} ${outputDir}"