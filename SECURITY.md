# Security Policy

## Supported versions

Only the latest release on `main` receives fixes.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private
[security advisory form](https://github.com/shreyescodes/vol-regime-backtester/security/advisories/new)
for this repository, and include steps to reproduce.

You can expect a first reply within a few days.

## Scope

QuantVol is a research tool that reads local files and, optionally, downloads public price data.
It does not place orders or handle credentials. Reports about unsafe file handling, for example
when loading CSV or Excel files or YAML configs, are in scope.
