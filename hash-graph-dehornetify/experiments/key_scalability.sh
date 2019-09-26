keycounts=($(seq 25 1 35))

execpath="../build"
resultsfile="./results/key_scalability.txt"

gpucount=2
bincount=16000
tablesize=($(echo "2^31 - 1" | bc))

for i in "${keycounts[@]}"
    do
        echo $i
        echo $i >> $resultsfile
        ./$execpath/multi-hash $i $tablesize $bincount $gpucount $bincount nocheck $i | grep "time" >> $resultsfile
    done
echo "" >> $resultsfile
#for i in 
#make multi-hash && ./multi-hash 16777216 16777216 16000 2 16000 check 16777216
