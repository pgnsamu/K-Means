#!/bin/bash
echo "select executable: "
echo "1) KMEANS_mpi"
echo "2) KMEANS_omp"
echo "3) KMEANS_cuda"
echo "4) KMEANS_mpi_omp"
echo "5) KMEANS_seq"
read execfile

read -p "input file name (es. input100D2): " inputfileName
read -p "number of clusers: " k
read -p "max iterations: " maxIter
read -p "min node muved %: " minMoved
read -p "distance precision: " precision
read -p "output file name: " outputfile

inputfile="../test_files/$inputfileName.inp"

outDirName="${inputfileName}_${k}_${maxIter}_${minMoved}_${precision}"

case $execfile in
  1)
    job="job/mpi.job"
    logDir="mpi/${outDirName}/logs"
    outputFile="mpi/${outDirName}/${outputfile}_out.txt"
    ;;
  2)
    job="job/omp.job"
    logDir="omp/${outDirName}/logs"
    outputFile="omp/${outDirName}/${outputfile}_out.txt"
    ;;
  3)
    job="job/cuda.job"
    logDir="cuda/${outDirName}/logs"
    outputFile="cuda/${outDirName}/${outputfile}_out.txt"
    ;;
  4)
    read -p "num thread: " threads
    export OMP_NUM_THREADS=$threads
    job="job/omp_mpi.job"
    logDir="omp_mpi/${outDirName}/logs"
    outputFile="omp_mpi/${outDirName}/${outputfile}_out.txt"
    ;;
  5)
    job="job/seq.job"
    logDir="../seq/${outDirName}/logs"
    outputFile="../seq/${outputfile}_out.txt"
    ;;
  *)
    echo "Invalid executable file"
    exit 1
    ;;
esac

make all

if [ ! -e "$inputfile" ]; then
  echo "$inputfile not found"
  exit 1
elif [ ! -d "$logDir" ]; then
  mkdir -p $logDir
elif [ "$k" -lt 1 ]; then
  echo "Invalid number of clusters"
  exit 1
elif [ "$maxIter" -lt 1 ]; then
  echo "Invalid number of max iterations"
  exit 1
elif [ "$minMoved" -lt 0 ]; then
  echo "Invalid min node muved %"
  exit 1
elif [ "$precision" -lt 0 ]; then
  echo "Invalid distance precision"
  exit 1
fi

condor_submit $job -append "log = ${logDir}/run_\$(Process).log" -append "output = ${logDir}/run_\$(Process).out" -append "error = ${logDir}/run_\$(Process).err" -append "arguments = ${inputfile} ${k} ${maxIter} ${minMoved} ${precision} ${outputFile}"
