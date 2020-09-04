"""
Discover and run tutorial scripts as unit tests.
"""

import os
import pytest
import subprocess

from pathlib import Path

# Define path to tutorial scripts
tutorials_dir = os.path.join(
    Path(__file__).resolve().parents[1],
    'tutorials', 'examples')

# Generate list of full path to scripts, and short display ids
scripts = []
display_ids = []
for filename in os.listdir(tutorials_dir):
    if filename.endswith('.py'):
        scripts.append(os.path.join(tutorials_dir, filename))
        display_ids.append(filename)


@pytest.mark.parametrize('filename', scripts, ids=display_ids)
def test_tutorials(filename):
    """
    Utility function to invoke script in a subprocess.

    Unsets DISPLAY.
    """

    # Acquire the environment, unset display for pyplot.
    env = os.environ
    env['DISPLAY'] = ''

    subprocess.check_call(['python', filename], env=env)
