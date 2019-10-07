echo "no index, no managed memory"

sed -i 's/^#define INDEX_TRACK/\/\/&/' ../include/MultiHashGraph.cuh
sed -i 's/^#define MANAGED_MEM/\/\/&/' ../include/MultiHashGraph.cuh

echo "strong scaling"
cd strong_scaling
./strong_scaling.sh ./results/ss_noindex_nomanaged.txt

echo "weak scaling"
cd ../weak_scaling
./weak_scaling.sh ./results/ws_noindex_nomanaged.txt

echo "duplicate keys"
cd ../duplicate_keys
./duplicate_keys.sh ./results/dk_noindex_nomanaged.txt


echo "index, no managed memory"

sed -i 's/^\/\/.*#define INDEX_TRACK/#define INDEX_TRACK/' ../include/MultiHashGraph.cuh
sed -i 's/^#define MANAGED_MEM/\/\/&/' ../include/MultiHashGraph.cuh

echo "strong scaling"
cd strong_scaling
./strong_scaling.sh ./results/ss_index_nomanaged.txt

echo "weak scaling"
cd ../weak_scaling
./weak_scaling.sh ./results/ws_index_nomanaged.txt

echo "duplicate keys"
cd ../duplicate_keys
./duplicate_keys.sh ./results/dk_index_nomanaged.txt


echo "no index, managed memory"

sed -i 's/^\/\/.*#define MANAGED_MEM/#define MANAGED_MEM/' ../include/MultiHashGraph.cuh
sed -i 's/^#define INDEX_TRACK/\/\/&/' ../include/MultiHashGraph.cuh

echo "strong scaling"
cd strong_scaling
./strong_scaling.sh ./results/ss_noindex_managed.txt

echo "weak scaling"
cd ../weak_scaling
./weak_scaling.sh ./results/ws_noindex_managed.txt

echo "duplicate keys"
cd ../duplicate_keys
./duplicate_keys.sh ./results/dk_noindex_managed.txt


echo "index, managed memory"

sed -i 's/^\/\/.*#define MANAGED_MEM/#define MANAGED_MEM/' ../include/MultiHashGraph.cuh
sed -i 's/^\/\/.*#define INDEX_TRACK/#define INDEX_TRACK/' ../include/MultiHashGraph.cuh

echo "strong scaling"
cd strong_scaling
./strong_scaling.sh ./results/ss_index_managed.txt

echo "weak scaling"
cd ../weak_scaling
./weak_scaling.sh ./results/ws_index_managed.txt

echo "duplicate keys"
cd ../duplicate_keys
./duplicate_keys.sh ./results/dk_index_managed.txt
