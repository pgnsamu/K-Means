#!/bin/bash
read -p "Enter the test file name: " testFile

for threads in 16; do
  logDir="mpi/${testFile}_${threads}/logs"
  mkdir -p "$logDir"
  
  condor_submit job/mpi.job \
    -append "TestFile = ${testFile}" \
    -append "Cpus = ${threads}"
done