#!/bin/bash


INPUT=$PWD/data/movies.tsv
OLDIFS=$IFS
IFS=','
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }

while read -r id url
do
	echo "Downoloading movie ID : $id"
	wget -nc -P $PWD/FUSION/data/LSMDC/$id -U "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)" $url
done < $INPUT
IFS=$OLDIFS

