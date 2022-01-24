for k in $(seq 11 1 12)
do
	for l in $(seq 25 10 200)
	do
		./pasha $k $l 16 ../k$k/decyc$k.txt ../k$k/hit$k@$l.txt > ../k$k/output$k@$l.txt
	done
done
