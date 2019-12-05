buildpath="../../build"
includepath="../../include"
sspath="./strong_scaling"
wspath="./weak_scaling"
dkpath="./duplicate_keys"

declare -a modes=("no index, no managed memory" "index, no managed memory" "no index, managed memory" "index, managed memory")

for mode in modes
  do
    bytes=8
    modefile="noindex_nomanaged"
    if [ "$mode" == "no index, no managed memory" ] then
      echo "no index, no managed memory"

      sed -i 's/^#define INDEX_TRACK/\/\/&/' $includepath/MultiHashGraph.cuh
      sed -i 's/^#define MANAGED_MEM/\/\/&/' $includepath/MultiHashGraph.cuh
    elif [ "$mode" == "index, no managed memory" ] then
      echo "index, no managed memory"

      sed -i 's/^\/\/.*#define INDEX_TRACK/#define INDEX_TRACK/' $includepath/MultiHashGraph.cuh
      sed -i 's/^#define MANAGED_MEM/\/\/&/' $includepath/MultiHashGraph.cuh
      bytes=16
      modefile="index_nomanaged"
    elif [ "$mode" == "no index, managed memory" ] then
      echo "index, no managed memory"

      sed -i 's/^\/\/.*#define INDEX_TRACK/#define INDEX_TRACK/' $includepath/MultiHashGraph.cuh
      sed -i 's/^#define MANAGED_MEM/\/\/&/' $includepath/MultiHashGraph.cuh
      modefile="noindex_managed"
    elif [ "$mode" == "index, managed memory" ] then
      echo "index, managed memory"

      sed -i 's/^\/\/.*#define MANAGED_MEM/#define MANAGED_MEM/' $includepath/MultiHashGraph.cuh
      sed -i 's/^\/\/.*#define INDEX_TRACK/#define INDEX_TRACK/' $includepath/MultiHashGraph.cuh
      bytes=16
      modefile="index_managed"
    fi

    make -C $buildpath multi-hash

    echo "strong scaling"
    jsrun -n 1 -g 6 $sspath/strong_scaling.sh $bytes > ./strong_scaling/results/ss_$modefile.txt
    head -n -$(cat ./strong_scaling/results/ss_$modefile.txt | grep -n intersect | cut -f1 -d:) ./strong_scaling/results/ss_$modefile.txt > ./strong_scaling/results/build/ss_$modefile.txt
    tail -n +$(cat ./strong_scaling/results/ss_$modefile.txt | grep -n intersect | cut -f1 -d:) ./strong_scaling/results/ss_$modefile.txt > ./strong_scaling/results/intersect/ss_$modefile.txt

    echo "weak scaling"
    jsrun -n 1 -g 6 $wspath/weak_scaling.sh $byte > ./weak_scaling/results/ws_$modefile.txt
    head -n -$(cat ./weak_scaling/results/ws_$modefile.txt | grep -n intersect | cut -f1 -d:) ./weak_scaling/results/ws_$modefile.txt > ./weak_scaling/results/build/ws_$modefile.txt
    tail -n +$(cat ./weak_scaling/results/ws_$modefile.txt | grep -n intersect | cut -f1 -d:) ./weak_scaling/results/ws_$modefile.txt > ./weak_scaling/results/intersect/ws_$modefile.txt

    echo "duplicate keys"
    jsrun -n 1 -g 6 $dkpath/duplicate_keys.sh $bytes > ./duplicate_keys/results/dk_$modefile.txt
    head -n -$(cat ./duplicate_keys/results/dk_$modefile.txt | grep -n intersect | cut -f1 -d:) ./duplicate_keys/results/dk_$modefile.txt > ./duplicate_keys/results/build/dk_$modefile.txt
    tail -n +$(cat ./duplicate_keys/results/dk_$modefile.txt | grep -n intersect | cut -f1 -d:) ./duplicate_keys/results/dk_$modefile.txt > ./duplicate_keys/results/intersect/dk_$modefile.txt

# echo "no index, no managed memory"
# sed -i 's/^#define INDEX_TRACK/\/\/&/' $includepath/MultiHashGraph.cuh
# sed -i 's/^#define MANAGED_MEM/\/\/&/' $includepath/MultiHashGraph.cuh
# 
# make -C $buildpath multi-hash
# 
# echo "strong scaling"
# jsrun -n 1 -g 6 $sspath/strong_scaling.sh 8 > ./strong_scaling/results/ss_noindex_nomanaged.txt
# 
# echo "weak scaling"
# jsrun -n 1 -g 6 $wspath/weak_scaling.sh > ./weak_scaling/results/ws_noindex_nomanaged.txt
# 
# echo "duplicate keys"
# jsrun -n 1 -g 6 $dkpath/duplicate_keys.sh > ./duplicate_keys/results/dk_noindex_nomanaged.txt
# 
# 
# echo "index, no managed memory"
# 
# sed -i 's/^\/\/.*#define INDEX_TRACK/#define INDEX_TRACK/' $includepath/MultiHashGraph.cuh
# sed -i 's/^#define MANAGED_MEM/\/\/&/' $includepath/MultiHashGraph.cuh
# 
# make -C $buildpath multi-hash
# 
# echo "strong scaling"
# jsrun -n 1 -g 6 $sspath/strong_scaling.sh 16 > ./strong_scaling/results/ss_index_nomanaged.txt
# 
# echo "weak scaling"
# jsrun -n 1 -g 6 $wspath/weak_scaling.sh > ./weak_scaling/results/ws_index_nomanaged.txt
# 
# echo "duplicate keys"
# jsrun -n 1 -g 6 $dkpath/duplicate_keys.sh > ./duplicate_keys/results/dk_index_nomanaged.txt
# 
# 
# echo "no index, managed memory"
# 
# sed -i 's/^\/\/.*#define MANAGED_MEM/#define MANAGED_MEM/' $includepath/MultiHashGraph.cuh
# sed -i 's/^#define INDEX_TRACK/\/\/&/' $includepath/MultiHashGraph.cuh
# 
# make -C $buildpath multi-hash
# 
# echo "strong scaling"
# jsrun -n 1 -g 6 $sspath/strong_scaling.sh 8 > ./strong_scaling/results/ss_noindex_managed.txt
# 
# echo "weak scaling"
# jsrun -n 1 -g 6 $wspath/weak_scaling.sh > ./weak_scaling/results/ws_noindex_managed.txt
# 
# echo "duplicate keys"
# jsrun -n 1 -g 6 $dkpath/duplicate_keys.sh > ./duplicate_keys/results/dk_noindex_managed.txt
# 
# 
# echo "index, managed memory"
# 
# sed -i 's/^\/\/.*#define MANAGED_MEM/#define MANAGED_MEM/' $includepath/MultiHashGraph.cuh
# sed -i 's/^\/\/.*#define INDEX_TRACK/#define INDEX_TRACK/' $includepath/MultiHashGraph.cuh
# 
# make -C $buildpath multi-hash
# 
# echo "strong scaling"
# jsrun -n 1 -g 6 $sspath/strong_scaling.sh 16 > ./strong_scaling/results/ss_index_managed.txt
# 
# echo "weak scaling"
# jsrun -n 1 -g 6 $wspath/weak_scaling.sh > ./weak_scaling/results/ws_index_managed.txt
# 
# echo "duplicate keys"
# jsrun -n 1 -g 6 $dkpath/duplicate_keys.sh > ./duplicate_keys/results/dk_index_managed.txt
