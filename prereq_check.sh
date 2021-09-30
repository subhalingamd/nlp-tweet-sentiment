FILES="writeup.txt compile-train.sh compile-test.sh run-train.sh run-test.sh main.py constants.py slang.txt preprocessing.py utils.py stats.py"

for FILE in $FILES;
do
	if [[ ! -f "$FILE" ]]; then
    	echo "$FILE missing!!"
    	exit 1;
	fi
done;
echo "All files present."