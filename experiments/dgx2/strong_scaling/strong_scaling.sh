keycounts=($(seq 24 1 33))
# gpucounts=($(seq 0 1 4))
gpucounts=(1 2 4 8 16)

execpath="../../build"
resultsfile=$1

bincount=16000

# sed -i 's/^\/\/.*#define INDEX_TRACK/#define INDEX_TRACK/' ../../include/MultiHashGraph.cuh
# make -C $execpath multi-hash

# rm $resultsfile
# echo "keycount,gpucount,time" >> $resultsfile
echo "strong_scaling"
echo "keycount,gpucount,time"
echo "build tests"
# echo "build tests" >> $resultsfile

# $1 is sizeof(keyval)
# $2 is 32-bit vs 64-bit
echo "countBinSizes,countKeyBuff,populateKeyBuffs,countFinalKeys,allToAll,building,total"
for i in "${keycounts[@]}"
    do
        let kc=$((echo 2^$i) | bc)
        # echo "keycount: ${kc}"
        for j in "${gpucounts[@]}"
            do
                # let gc=$((echo 2^$j) | bc)
                let gc=$j
                # echo "gpucount: ${gc}"

                let ts=$kc
                if [ $2 -eq 4 ] ; then
                    ts=$(($kc < (2**($2 * 8 - 1)) ? $kc : 2**($2 * 8 - 1)))
                fi

                # internal cuda malloc + keys + hashes + keyBinBuff
                let gigs=$((echo "((($kc * $1) + ($kc * $2) + (2 * $kc * $1) + (2 * $kc * 8)) + ($kc * $2) + ($kc * $2) + ($kc * $1)) / 2^30") | bc)
                let gpureq=$((echo "($gigs + 32) / 32") | bc)

                if (( $gpureq > $gc )) ; then
                  echo "${kc},${gc},oom"
                  continue
                else 
                  # ans=$(./$execpath/multi-hash $kc $kc $bincount $gc $bincount nocheck $kc build | grep "time")
                  ans=$(./$execpath/multi-hash $kc $ts $bincount $gc $bincount nocheck $kc build | grep "time")
                  # tokens=( $ans )
                  # time=${tokens[3]}

                  # echo "${kc},${gc},${time}" >> $resultsfile
                  # echo "${kc},${gc},${time}"
                  echo -e "${kc},${gc},\n${ans}"
                  break
                fi
            done
    done

echo "intersect tests"
# echo "intersect tests" >> $resultsfile
for i in "${keycounts[@]}"
    do
        let kc=$((echo 2^$i) | bc)
        kc=$((kc / 2))
        # echo "keycount: ${kc}"
        for j in "${gpucounts[@]}"
            do
                # let gc=$((echo 2^$j) | bc)
                let gc=$j
                # echo "gpucount: ${gc}"

                let ts=$kc
                if [ $2 -eq 4 ] ; then
                    ts=$(($kc < (2**($2 * 8 - 1)) ? $kc : 2**($2 * 8 - 1)))
                fi

                # internal cuda malloc + keys + hashes + keyBinBuff
                let gigs=$((echo "((($kc * $1) + ($kc * $2) + (2 * $kc * $1) + (2 * $kc * 8)) + ($kc * $2) + ($kc * $2) + ($kc * $1)) / 2^30") | bc)
                let gpureq=$((echo "($gigs * 2 + 32) / 32") | bc)

                if (( $gpureq > $gc )) ; then
                  echo "${kc},${gc},oom"
                  continue
                else
                  # ans=$(./$execpath/multi-hash $kc $kc $bincount $gc $bincount nocheck $kc intersect | grep "time")
                  ans=$(./$execpath/multi-hash $kc $ts $bincount $gc $bincount nocheck $kc intersect | grep "time")
                  # tokens=( $ans )
                  # time=${tokens[3]}

                  # echo "${kc},${gc},${time}" >> $resultsfile
                  # echo "${kc},${gc},${time}"
                  echo -e "${kc},${gc},\n${ans}"
                  break
                fi
            done
    done

