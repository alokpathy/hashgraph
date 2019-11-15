includepath="../../include"
sspath="./strong_scaling"
wspath="./weak_scaling"
dkpath="./duplicate_keys"

echo "no index, no managed memory"
 
sed -i 's/^#define INDEX_TRACK/\/\/&/' $includepath/MultiHashGraph.cuh
sed -i 's/^#define MANAGED_MEM/\/\/&/' $includepath/MultiHashGraph.cuh

echo "strong scaling"
$sspath/strong_scaling.sh $sspath/results/ss_noindex_nomanaged.txt

echo "weak scaling"
$wspath/weak_scaling.sh $wspath/results/ws_noindex_nomanaged.txt

echo "duplicate keys"
$dkpath/duplicate_keys.sh $dkpath/results/dk_noindex_nomanaged.txt


echo "index, no managed memory"

sed -i 's/^\/\/.*#define INDEX_TRACK/#define INDEX_TRACK/' $includepath/MultiHashGraph.cuh
sed -i 's/^#define MANAGED_MEM/\/\/&/' $includepath/MultiHashGraph.cuh

echo "strong scaling"
$sspath/strong_scaling.sh $sspath/results/ss_index_nomanaged.txt

echo "weak scaling"
$wspath/weak_scaling.sh $wspath/results/ws_index_nomanaged.txt

echo "duplicate keys"
$dkpath/duplicate_keys.sh $dkpath/results/dk_index_nomanaged.txt


echo "no index, managed memory"

sed -i 's/^\/\/.*#define MANAGED_MEM/#define MANAGED_MEM/' $includepath/MultiHashGraph.cuh
sed -i 's/^#define INDEX_TRACK/\/\/&/' $includepath/MultiHashGraph.cuh

echo "strong scaling"
$sspath/strong_scaling.sh $sspath/results/ss_noindex_managed.txt

echo "weak scaling"
$wspath/weak_scaling.sh $wspath/results/ws_noindex_managed.txt

echo "duplicate keys"
$dkpath/duplicate_keys.sh $dkpath/results/dk_noindex_managed.txt


echo "index, managed memory"

sed -i 's/^\/\/.*#define MANAGED_MEM/#define MANAGED_MEM/' $includepath/MultiHashGraph.cuh
sed -i 's/^\/\/.*#define INDEX_TRACK/#define INDEX_TRACK/' $includepath/MultiHashGraph.cuh

echo "strong scaling"
$sspath/strong_scaling.sh $sspath/results/ss_index_managed.txt

echo "weak scaling"
$wspath/weak_scaling.sh $wspath/results/ws_index_managed.txt

echo "duplicate keys"
$dkpath/duplicate_keys.sh $dkpath/results/dk_index_managed.txt
