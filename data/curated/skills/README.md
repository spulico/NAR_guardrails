# Test fixtures: do not install

Half of these folders are deliberately malicious agent skills (exfiltration, credential
harvesting, persistence, tool poisoning and more). They exist to test skill scanners and must
never be installed, loaded or executed by an agent. Labels are in `labels.jsonl`; the whole set
in one file is `../skills.jsonl`. See `../README.md`.
