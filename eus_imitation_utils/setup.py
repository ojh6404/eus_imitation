from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

## DO NOT run this code manually !!!

setup_args = generate_distutils_setup(
  packages=['eus_imitation_utils'],
  package_dir={'': 'src'},
)

setup(**setup_args)
