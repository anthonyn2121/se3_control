import os
import sys

# Get the absolute path of the 'environment_toolkit' submodule
submodule_path = os.path.join(os.path.dirname(__file__), 'environment_toolkit')

# Add the submodule's path to the Python path
sys.path.append(submodule_path)

# Get the absolute path of the 'environment_toolkit' submodule
# submodule_path = os.path.join(os.path.dirname(__file__), 'a_star')

# Add the submodule's path to the Python path
# sys.path.append(submodule_path)

# Get the absolute path of the 'environment_toolkit' submodule
submodule_path = os.path.join(os.path.dirname(__file__), 'trajectory_generation')

# Add the submodule's path to the Python path
sys.path.append(submodule_path)