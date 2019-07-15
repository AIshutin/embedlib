!/bin/bash

FILE=./ubuntu_dialogs.tgz
if [ ! -f "$FILE" ]; then
     echo "$FILE does not exist -> downloading"
     wget dataset.cs.mcgill.ca/ubuntu-corpus-1.0/ubuntu_dialogs.tgz
     tar zxvf ubuntu_dialogs.tgz >/dev/null
fi
