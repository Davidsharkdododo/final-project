#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch publisher
rosrun my_package dot.py

# wait for app to end
dt-launchfile-join