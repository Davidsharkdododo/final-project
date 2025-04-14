#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------

# Retrieve the custom parameter value from the environment variable
# Provide a default value if the environment variable is not set (good practice)
MY_PARAM=${MY_CUSTOM_PARAM_VALUE:-"default_value"}

# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)

# launching app
# Add the new parameter 'my_custom_param' to the roslaunch command
dt-exec roslaunch my_package final.launch veh:=$VEHICLE_NAME my_custom_param:=$MY_PARAM


# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join