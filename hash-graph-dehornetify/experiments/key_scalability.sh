keycounts=($(seq 25 1 34))

execpath="../build"
resultsfile="./results/key_scalability.txt"

gpucount=16
bincount=16000
#tablesize=($(echo "2^31 - 1" | bc))

echo "build tests"
echo "build tests" >> $resultsfile
for i in "${keycounts[@]}"
    do
        echo $i
        echo $i >> $resultsfile
        # ./$execpath/multi-hash $i $tablesize $bincount $gpucount $bincount nocheck $i build | grep "time" >> $resultsfile
        ./$execpath/multi-hash $(echo "2^$i" | bc) $(echo "2^$i" | bc)  $bincount $gpucount $bincount nocheck $(echo "2^$i" | bc)  build | grep "time" >> $resultsfile
    done
echo "" >> $resultsfile

echo "intersect tests"
echo "intersect tests" >> $resultsfile

# $tablesize=($(echo "2^31 - 1" | bc ))
for i in "${keycounts[@]}"
    do
        echo $i
        echo $i >> $resultsfile
        # ./$execpath/multi-hash $(echo "2^33" | bc) $tablesize $bincount $gpucount $bincount nocheck $i intersect | grep "time" >> $resultsfile
        ./$execpath/multi-hash $(echo "2^$i" | bc) $(echo "2^$i" | bc)  $bincount $gpucount $bincount nocheck $(echo "2^$i" | bc)  intersect | grep "time" >> $resultsfile
    done
echo "" >> $resultsfile
