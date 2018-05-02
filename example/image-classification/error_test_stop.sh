#!/bin/bash
PID="`ps -ef|grep train_|grep -v 'grep'|awk '{print $2}' ORS=","`"
echo $PID | awk '{split($0,arr,",");cmd="kill -9 "; for(i in arr) system(cmd arr[i])}'


