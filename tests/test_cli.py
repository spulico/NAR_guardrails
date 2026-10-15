import subprocess, sys

def test_cli_help_runs():
    from nar_guardrails.cli import main
    assert main(["--help"]) == 0

def test_package_imports():
    import nar_guardrails
    assert nar_guardrails.__version__ == "0.1.0"
