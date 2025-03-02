#!/bin/sh

trap "kill 0" SIGINT

taskset -c 60 python Step_2_Train_MSAD.py -t ID & 
taskset -c 61 python Step_2_Train_MSAD.py -t WebService & 
taskset -c 62 python Step_2_Train_MSAD.py -t Medical & 
taskset -c 63 python Step_2_Train_MSAD.py -t Facility & 
taskset -c 64 python Step_2_Train_MSAD.py -t Synthetic & 
taskset -c 65 python Step_2_Train_MSAD.py -t HumanActivity & 
taskset -c 66 python Step_2_Train_MSAD.py -t Sensor & 
taskset -c 67 python Step_2_Train_MSAD.py -t Environment & 
taskset -c 68 python Step_2_Train_MSAD.py -t Finance & 
taskset -c 69 python Step_2_Train_MSAD.py -t Traffic & 

# taskset -c 20 python Step_2_Train_MetaOD.py -t ID & 
# taskset -c 21 python Step_2_Train_MetaOD.py -t WebService & 
# taskset -c 22 python Step_2_Train_MetaOD.py -t Medical & 
# taskset -c 23 python Step_2_Train_MetaOD.py -t Facility & 
# taskset -c 24 python Step_2_Train_MetaOD.py -t Synthetic & 
# taskset -c 25 python Step_2_Train_MetaOD.py -t HumanActivity & 
# taskset -c 26 python Step_2_Train_MetaOD.py -t Sensor & 
# taskset -c 27 python Step_2_Train_MetaOD.py -t Environment & 
# taskset -c 28 python Step_2_Train_MetaOD.py -t Finance & 
# taskset -c 19 python Step_2_Train_MetaOD.py -t Traffic & 
