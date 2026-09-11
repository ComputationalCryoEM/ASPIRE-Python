"""
Tests for the CLI wrapper
"""

import subprocess


def test_help_menu():
    """Tests the cli wrapper can actually run.

    While `click` provides `CliRunner` to test commands/option to the
    cli, this is more straight forward. Just ensure the thing doesn't
    crash when a user invokes `aspire` in their shell.
    """

    subprocess.run(["aspire", "--help"], check=True, timeout=30)
