find "../task10ABCD_cleansed_4_scoring/" -name "*-A.output" \
     -exec perl score-semeval2015-task10-subtaskA.pl {} \; \
     | perl -p -e 's/^.+\/([^\/]+)\/[^\/]+-A\.output/$1/g' \
     | sort -r -n -k3 > ../results/results-A.txt

find "../task10ABCD_cleansed_4_scoring/" -name "*-B.output" \
	 -exec perl score-semeval2015-task10-subtaskB.pl {} \; \
     | perl -p -e 's/^.+\/([^\/]+)\/[^\/]+-B\.output/$1/g' \
     | sort -r -n -k3 > ../results/results-B.txt

find "../task10ABCD_cleansed_4_scoring/" -name "*-C.output" \
	 -exec perl score-semeval2015-task10-subtaskC.pl {} \; \
     | perl -p -e 's/^.+\/([^\/]+)\/[^\/]+-C\.output/$1/g' \
     | sort -r -n -k3 > ../results/results-C.txt

find "../task10ABCD_cleansed_4_scoring/" -name "*-D.output" \
     -exec perl score-semeval2015-task10-subtaskD.pl {} \; \
     | perl -p -e 's/^.+\/([^\/]+)\/[^\/]+-D\.output/$1/g' \
     | sort -n -k3 > ../results/results-D.txt


find "../task10ABCD_cleansed_4_scoring/" -name "*A-progress*.output" \
	 -exec perl score-semeval2014-task9-subtaskA.pl {} \; \
     | perl -p -e 's/^.+\/([^\/]+)\/[^\/]+-A-progress\.output/$1/g' \
     | sort -r -n -k9 > ../results/results-A-progress.txt

find "../task10ABCD_cleansed_4_scoring/" -name "*B-progress*.output" \
	 -exec perl score-semeval2014-task9-subtaskB.pl {} \; \
     | perl -p -e 's/^.+\/([^\/]+)\/[^\/]+-B-progress\.output/$1/g' \
     | sort -r -n -k9 > ../results/results-B-progress.txt
