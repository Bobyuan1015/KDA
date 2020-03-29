#!/usr/bin/env bash

rm -rf log.txt experiments.csv checkpoints/*

tasks="run_cnn.py run_rnn.py"
tasks="run_rnn.py"
names="weibo_senti_100k chnsenticorp cnews_10"
names="chnsenticorp"

numbers=10
numbers=2

data_size="data_500 data_2000 data_orginal data_500_kda_5000 data_2000_kda_5000 data_500_kda_10000 data_2000_kda_10000 data_500_copy_5000 data_2000_copy_5000 data_500_copy_10000 data_2000_copy_10000 data_500_eda_5000 data_2000_eda_5000 data_500_eda_10000 data_2000_eda_10000"
#data_size="data_orginal"

for name in $names;
  do
    for data in $data_size;
      do
          if [ $name = "cnews_10" ];then
             class='体育-财经-房产-家居-教育-科技-时尚-时政-游戏-娱乐'
            echo $class
          fi
          if [ $name = "weibo_senti_100k" ];then
            class='0-1'
            echo $class
          fi
          if [ $name = "chnsenticorp" ];then
            class="0-1"
            echo $class
          fi

        for task in $tasks;
          do
              echo "run "$task >> log.txt
              for((i=1;i<$numbers;i++))
                do
                  time=$(date "+%Y%m%d-%Hh%Mm%Ss")
                  echo 'start  at '$time', Run the '$i'th '$task'  on                                                   --'$data >> log.txt
                  python $task train $i $name $data $class>> log.txt
                  python $task test $i $name $data $class>> log.txt
#                  python $task train $i $name $data $class
#                  python $task test $i $name $data $class
                  time=$(date "+%Y%m%d-%Hh%Mm%Ss")
                  echo 'end    at '$time'      the '$i'th '$task'  running' >> log.txt
                done
          done
      done
  done