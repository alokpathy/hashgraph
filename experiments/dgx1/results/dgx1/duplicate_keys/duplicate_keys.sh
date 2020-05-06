keycount=26
gpucount=8
tablesizes=($(seq 21 1 31))

execpath="../../build"
resultsfile=$1

bincount=16000

# sed -i 's/^\/\/.*#define INDEX_TRACK/#define INDEX_TRACK/' ../../include/MultiHashGraph.cuh
# make -C $execpath multi-hash

rm $resultsfile
echo "keycount,tablesize,gpucount,time" >> $resultsfile
echo "duplicate_keys"
echo "keycount,tablesize,gpucount,time"
echo "build tests"
echo "build tests" >> $resultsfile

for i in "${tablesizes[@]}"
    do
        let kc=$((echo 2^$keycount) | bc)
        # let gc=$((echo 2^$gpucount) | bc)
        let gc=$gpucount
        let ts=$((echo 2^$i) | bc)

        echo "tableSize: ${ts}"
        ans=$(./$execpath/multi-hash $kc $ts $bincount $gc $bincount nocheck $kc build | grep "time")

        tokens=( $ans )
        time=${tokens[3]}

        echo "${kc},${ts},${gc},${time}" >> $resultsfile
    done

echo "intersect tests"
echo "intersect tests" >> $resultsfile

keycount=$((echo $keycount - 1) | bc)
for i in "${tablesizes[@]}"
    do
        let kc=$((echo 2^$keycount) | bc)
        # let gc=$((echo 2^$gpucount) | bc)
        let gc=$gpucount
        let ts=$((echo 2^$i) | bc)

        echo "tableSize: ${ts}"
        ans=$(./$execpath/multi-hash $kc $ts $bincount $gc $bincount nocheck $kc intersect | grep "time")

        tokens=( $ans )
        time=${tokens[3]}

        echo "${kc},${ts},${gc},${time}" >> $resultsfile
    done
