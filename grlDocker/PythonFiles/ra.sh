#!/bin/bash
echo -en "\033[1;32m --- Attach --- \033[0m\n"

python ./PythonFiles/leo_test.py &
python ./PythonFiles/main_ddpg.py 


