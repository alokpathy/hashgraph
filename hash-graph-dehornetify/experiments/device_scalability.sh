gpucounts=($(seq 12 1 16))

execpath="../build"
resultsfile="./results/device_scalability.txt"

keycount=($(echo "2^33"))
bincount=16000
tablesize=($(echo "2^31 - 1" | bc))

echo "build tests"
echo "build tests" >> $resultsfile
for i in "${gpucounts[@]}"
    do
        echo $i
        echo $i >> $resultsfile
        ./$execpath/multi-hash $keycount $tablesize $bincount $i $bincount nocheck $keycount build | grep "time" >> $resultsfile
    done
echo "" >> $resultsfile

echo "intersect tests"
echo "intersect tests" >> $resultsfile

for i in "${gpucounts[@]}"
    do
        echo $i
        echo $i >> $resultsfile
        ./$execpath/multi-hash $keycount $tablesize $bincount $i $bincount nocheck $keycount intersect | grep "time" >> $resultsfile
    done
echo "" >> $resultsfile
