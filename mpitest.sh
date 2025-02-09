#!/bin/bash
read -p "Enter the test file name: " testFile

for threads in 16;
do
  for run in $(seq 1 100)
  do
    logDir="mpi/${testFile}_${threads}/logs"
    job="job/mpi.job"
    inputfile="test_files/${testFile}.inp"
    outputFile="mpi/${testFile}_${threads}/output.txt"
    mkdir -p $logDir
    condor_submit $job -append "log = ${logDir}/run.log" -append "output = ${logDir}/run_${run}_\$(NODE).out" -append "error = ${logDir}/run_${run}_\$(NODE).err" -append "arguments = KMEANS_mpi ${inputfile} 128 1000 1 1 ${outputFile}" -append "request_cpus = ${threads}" -append "transfer_input_files = test_files, KMEANS_mpi, mpi/${testFile}_${threads}"
  done
done