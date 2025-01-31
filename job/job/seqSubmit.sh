#!/bin/bash
runs=$1 
logDir=$2 
inputDir=$3 
k=$4 
maxIter=$5 
minMoved=$6 
precision=$7 
outputFile=$8
extractedTime=$9

echo "Starting $runs runs"
echo "Args:  $runs $logDir $inputDir $k $maxIter $minMoved $precision $outputFile"

for i in $(seq 1 $runs)
do
   echo "Run $i"
   ./KMEANS_seq $inputDir $k $maxIter $minMoved $precision $outputFile 1>${logDir}/run_$i.log 2>${logDir}/run_$i.err
done

echo "All runs completed"
python3 dataExtraction.py "${logDir}/" $extractedTime
echo "Data extraction completed"