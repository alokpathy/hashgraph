keycounts=($(seq 24 1 33))
# gpucounts=($(seq 0 1 4))
gpucounts=(1 2 4 6)
# gpucounts=(1 2 4 8)

execpath="../../build"
resultsfile=$1

bincount=16000

# sed -i 's/^\/\/.*#define INDEX_TRACK/#define INDEX_TRACK/' ../../include/MultiHashGraph.cuh
# make -C $execpath multi-hash

# rm $resultsfile
# echo "keycount,gpucount,time" >> $resultsfile
echo "weak_scaling"
echo "keycount,gpucount,time"
echo "build tests"
# echo "build tests" >> $resultsfile

echo "countBinSizes,countKeyBuff,populateKeyBuffs,countFinalKeys,allToAll,building,total"
for i in "${keycounts[@]}"
    do
        let kcdev=$((echo 2^$i) | bc)
        # echo "keycount / dev: ${kcdev}"
        for j in "${gpucounts[@]}"
            do
                # let gc=$((echo 2^$j) | bc)
                let gc=$j
                # echo "gpucount: ${gc}"

                let kc=$(($kcdev * $gc))

                # internal cuda malloc + keys + hashes + keyBinBuff
                let gigs=$((echo "((($kc * $1) + ($kc * $2) + (2 * $kc * $1) + (2 * $kc * 8)) + ($kc * $2) + ($kc * $2) + ($kc * $1)) / 2^30") | bc)
                let gpureq=$((echo "($gigs + 16) / 16") | bc)

                if (( $gpureq > $gc )) ; then
                  echo "${kc},${gc},oom"
                  continue
                fi

                ans=$(./$execpath/multi-hash $kc $kc $bincount $gc $bincount nocheck $kc build | grep "time")
                # tokens=( $ans )
                # time=${tokens[3]}

                # echo "${kc},${gc},${time}" >> $resultsfile
                echo -e "${kc},${gc},\n${ans}"
            done
    done

echo "intersect tests"
# echo "intersect tests" >> $resultsfile

for i in "${keycounts[@]}"
    do
        let kcdev=$((echo 2^$i) | bc)
        kcdev=$((kcdev / 2))
        # echo "keycount / dev : ${kcdev}"
        for j in "${gpucounts[@]}"
            do
                # let gc=$((echo 2^$j) | bc)
                let gc=$j
                # echo "gpucount: ${gc}"

                let kc=$(($kcdev * $gc))

                # internal cuda malloc + keys + hashes + keyBinBuff
                let gigs=$((echo "((($kc * $1) + ($kc * $2) + (2 * $kc * $1) + (2 * $kc * 8)) + ($kc * $2) + ($kc * $2) + ($kc * $1)) / 2^30") | bc)
                let gpureq=$((echo "($gigs * 2 + 16) / 16") | bc)

                if (( $gpureq > $gc )) ; then
                  echo "${kc},${gc},oom"
                  continue
                fi

                ans=$(./$execpath/multi-hash $kc $kc $bincount $gc $bincount nocheck $kc intersect | grep "time")
                # tokens=( $ans )
                # time=${tokens[3]}

                # echo "${kc},${gc},${time}" >> $resultsfile
                echo -e "${kc},${gc},\n${ans}"
            done
    done

