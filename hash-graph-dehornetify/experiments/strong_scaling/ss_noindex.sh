keycounts=($(seq 24 1 33))
gpucounts=($(seq 0 1 4))

execpath="../../build"
resultsfile="./results/ss_noindex_results.txt"

bincount=16000

sed -i 's/^#define INDEX_TRACK/\/\/&/' ../../include/MultiHashGraph.cuh
make -C $execpath multi-hash

rm $resultsfile
echo "keycount,gpucount,time" >> $resultsfile
echo "build tests"
echo "build tests" >> $resultsfile

for i in "${keycounts[@]}"
    do
        let kc=$((echo 2^$i) | bc)
        echo "keycount: ${kc}"
        for j in "${gpucounts[@]}"
            do
                let gc=$((echo 2^$j) | bc)
                echo "gpucount: ${gc}"

                ans=$(./$execpath/multi-hash $kc $kc $bincount $gc $bincount nocheck $kc build | grep "time")
                tokens=( $ans )
                time=${tokens[3]}

                echo "${kc},${gc},${time}" >> $resultsfile
            done
    done

echo "intersect tests"
echo "intersect tests" >> $resultsfile

for i in "${keycounts[@]}"
    do
        let kc=$((echo 2^$i) | bc)
        kc=$((kc / 2))
        echo "keycount: ${kc}"
        for j in "${gpucounts[@]}"
            do
                let gc=$((echo 2^$j) | bc)
                echo "gpucount: ${gc}"

                ans=$(./$execpath/multi-hash $kc $kc $bincount $gc $bincount nocheck $kc intersect | grep "time")
                tokens=( $ans )
                time=${tokens[3]}

                echo "${kc},${gc},${time}" >> $resultsfile
            done
    done

