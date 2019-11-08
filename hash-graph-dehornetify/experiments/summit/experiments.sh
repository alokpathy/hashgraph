# echo "no index, no managed memory"
# 
sed -i 's/^#define INDEX_TRACK/\/\/&/' ../include/MultiHashGraph.cuh
sed -i 's/^#define MANAGED_MEM/\/\/&/' ../include/MultiHashGraph.cuh

echo "strong scaling"
jsrun -n 1 -g 6 ./strong_scaling/strong_scaling.sh # ./strong_scaling/results/ss_noindex_nomanaged.txt

echo "weak scaling"
jsrun -n 1 -g 6 ./weak_scaling/weak_scaling.sh # ./weak_scaling/results/ws_noindex_nomanaged.txt

echo "duplicate keys"
jsrun -n 1 -g 6 ./duplicate_keys/duplicate_keys.sh # ./duplicate_keys/results/dk_noindex_nomanaged.txt


echo "index, no managed memory"

sed -i 's/^\/\/.*#define INDEX_TRACK/#define INDEX_TRACK/' ../include/MultiHashGraph.cuh
sed -i 's/^#define MANAGED_MEM/\/\/&/' ../include/MultiHashGraph.cuh

echo "strong scaling"
jsrun -n 1 -g 6 ./strong_scaling/strong_scaling.sh # ./strong_scaling/results/ss_index_nomanaged.txt

echo "weak scaling"
jsrun -n 1 -g 6 ./weak_scaling/weak_scaling.sh # ./weak_scaling/results/ws_index_nomanaged.txt

echo "duplicate keys"
jsrun -n 1 -g 6 ./duplicate_keys/duplicate_keys.sh # ./duplicate_keys/results/dk_index_nomanaged.txt


echo "no index, managed memory"

sed -i 's/^\/\/.*#define MANAGED_MEM/#define MANAGED_MEM/' ../include/MultiHashGraph.cuh
sed -i 's/^#define INDEX_TRACK/\/\/&/' ../include/MultiHashGraph.cuh

echo "strong scaling"
jsrun -n 1 -g 6 ./strong_scaling/strong_scaling.sh # ./strong_scaling/results/ss_noindex_managed.txt

echo "weak scaling"
jsrun -n 1 -g 6 ./weak_scaling/weak_scaling.sh # ./weak_scaling/results/ws_noindex_managed.txt

echo "duplicate keys"
jsrun -n 1 -g 6 ./duplicate_keys/duplicate_keys.sh # ./duplicate_keys/results/dk_noindex_managed.txt


echo "index, managed memory"

sed -i 's/^\/\/.*#define MANAGED_MEM/#define MANAGED_MEM/' ../include/MultiHashGraph.cuh
sed -i 's/^\/\/.*#define INDEX_TRACK/#define INDEX_TRACK/' ../include/MultiHashGraph.cuh

echo "strong scaling"
jsrun -n 1 -g 6 ./strong_scaling/strong_scaling.sh # ./strong_scaling/results/ss_index_managed.txt

echo "weak scaling"
jsrun -n 1 -g 6 ./weak_scaling/weak_scaling.sh # ./weak_scaling/results/ws_index_managed.txt

echo "duplicate keys"
jsrun -n 1 -g 6 ./duplicate_keys/duplicate_keys.sh # ./duplicate_keys/results/dk_index_managed.txt
