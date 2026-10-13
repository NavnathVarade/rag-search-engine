# Security Policy

## Reporting a vulnerability
Please open a private security advisory (or email the maintainers) rather
than a public issue for anything that could allow unauthorized data
access, SSRF, injection, or authentication bypass.

## Scope and known limitations
See [docs/security.md](docs/security.md) "Known gaps" for an explicit,
honest list of what is and isn't implemented (e.g. production
authentication middleware is not wired in this reference build —
`AUTH_ENABLED=true` should not be used until that exists).

## Supported versions
This is a reference/educational implementation (0.x); there is no formal
LTS/support window at this stage.
