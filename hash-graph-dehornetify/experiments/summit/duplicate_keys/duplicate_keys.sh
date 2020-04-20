keycount=29
gpucount=6
tablesizes=($(seq 20 1 30))

execpath="../../build"
resultsfile=$1

bincount=16000

# sed -i 's/^\/\/.*#define INDEX_TRACK/#define INDEX_TRACK/' ../../include/MultiHashGraph.cuh
# make -C $execpath multi-hash

# rm $resultsfile
# echo "keycount,tablesize,gpucount,time" >> $resultsfile
echo "duplicate_keys"
echo "keycount,tablesize,gpucount,time"
echo "build tests"
# echo "build tests" >> $resultsfile

for i in "${tablesizes[@]}"
    do
        let kc=$((echo 2^$keycount) | bc)
        # let gc=$((echo 2^$gpucount) | bc)
        let gc=$gpucount
        let ts=$((echo 2^$i) | bc)

        # echo "tableSize: ${ts}"
        # internal cuda malloc + keys + hashes + keyBinBuff + temp space
        let gigs=$((echo "((($kc * $1) + ($kc * 8) + (2 * $kc * $1) + (2 * $ts * 8)) + ($kc * 8) + ($kc * 8) + ($kc * $1) + ($ts * 8)) / 2^30") | bc)
        let gpureq=$((echo "($gigs + 16) / 16") | bc)

        if (( $gpureq > $gc )) ; then
          echo "${kc},${ts},${gc},oom"
          continue
        fi

        ans=$(./$execpath/multi-hash $kc $ts $bincount $gc $bincount nocheck $kc build | grep "time")

        tokens=( $ans )
        time=${tokens[3]}

        # echo "${kc},${ts},${gc},${time}" >> $resultsfile
        echo "${kc},${ts},${gc},${time}"
    done

echo "intersect tests"
# echo "intersect tests" >> $resultsfile

keycount=$((echo $keycount - 1) | bc)
for i in "${tablesizes[@]}"
    do
        let kc=$((echo 2^$keycount) | bc)
        # let gc=$((echo 2^$gpucount) | bc)
        let gc=$gpucount
        let ts=$((echo 2^$i) | bc)

        # echo "tableSize: ${ts}"
        # internal cuda malloc + keys + hashes + keyBinBuff + temp space
        let gigs=$((echo "((($kc * $1) + ($kc * 8) + (2 * $kc * $1) + (2 * $ts * 8)) + ($kc * 8) + ($kc * 8) + ($kc * $1) + ($ts * 8)) / 2^30") | bc)
        let gpureq=$((echo "($gigs * 2 + 16) / 16") | bc)

        if (( $gpureq > $gc )) ; then
          echo "${kc},${ts},${gc},oom"
          continue
        fi
        ans=$(./$execpath/multi-hash $kc $ts $bincount $gc $bincount nocheck $kc intersect | grep "time")

        tokens=( $ans )
        time=${tokens[3]}

        # echo "${kc},${ts},${gc},${time}" >> $resultsfile
        echo "${kc},${ts},${gc},${time}"
    done
