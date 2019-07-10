#!/bin/bash
# Copyright FUJITSU LIMITED 2019

if [ -d $1 ] ; then
    LogDir=${1}
else
    LogDir=${1%/*}
fi

FileStdOut=$LogDir/stdout.txt
FileStdErr=$LogDir/stderr.txt

if [ ! -f $FileStdOut -o ! -f $FileStdErr ] ; then
    echo stdout.txt or stderr.txt is not exist!
    exit
fi

JobName=`echo $FileStdOut | cut -d '/' -f 3`

SpeedMin=`grep Speed $FileStdErr | sort -n -k 5 | head -n 2 | tail -n 1 | cut -d ' ' -f 4`
SpeedMax=`grep Speed $FileStdErr | sort -n -k 5 | tail -n 1 | cut -d ' ' -f 4`

if [ -z $SpeedMin ]; then
    SpeedMin="-"
fi
if [ -z $SpeedMax ]; then
    SpeedMax="-"
fi

MLPv050=`grep "MLPv0.5.0" $FileStdOut | tail -n 1`

if [ -z $MLPv050 ]; then
    # for MLPerf v0.6
    EpochAndAcc=`awk '/eval_accuracy/{if (match($0, "\"value\"[ :]*([0-9.]+)", arr)) { value = arr[1] } if (match($0, "\"epoch_num\"[ :]*([0-9.]+)", arr)) { epoch = arr[1] } print epoch, value}' $FileStdOut | sort -n -k 2 | tail -n 1`
    Acc=`echo $EpochAndAcc | cut -d ' ' -f 2`
    AccEpoch=`echo $EpochAndAcc | cut -d ' ' -f 1`

    TimeRunStart=`grep run_start $FileStdOut | cut -d ' ' -f 2`
    TimeRunStop=`grep run_stop $FileStdOut | cut -d ' ' -f 2`
else
    # for MLPerf v0.5
    Acc=`grep eval_accuracy $FileStdErr | sort -n -k 9 | tail -n 1 | cut -d ' ' -f 9 | sed 's/}//'`
    AccEpoch=`grep eval_accuracy $FileStdErr | sort -n -k 9 | tail -n 1 | cut -d ' ' -f 7 | sed 's/,//'`

    TimeRunStart=`grep run_start $FileStdErr | cut -d ' ' -f 3`
    TimeRunStop=`grep run_stop $FileStdErr | cut -d ' ' -f 3`
fi

#echo ""
#echo "---Parameters---"

echo "### Time (run_start to run_stop) ### "
echo "$TimeRunStop - $TimeRunStart" | bc
echo "### JobName SpeedMin SpeedMax - Acc Epoch TimeRunStart TimeRunStop ###"
echo $JobName $SpeedMax $SpeedMin - $Acc $AccEpoch $TimeRunStart $TimeRunStop

