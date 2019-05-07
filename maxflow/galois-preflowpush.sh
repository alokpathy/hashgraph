#!/bin/sh
#Wrapper for galois maxflow

if [ "$#" -lt "3" ]
then
	echo "Usage : $0 <matrix gr format> <source> <target>"
	exit 0
fi

s=$(($2 - 1)) #base 0 in galois
t=$(($3 - 1))

#Moving to own directory

DIR="$( cd "$(dirname "$0")" && pwd )"
$DIR/galois/build/apps/preflowpush/preflowpush "$1" "$s" "$t" -t 12 --noverify | grep "^time\|^max"
