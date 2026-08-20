# Security Policy

## Supported versions

| Version | Supported |
| --- | --- |
| 0.4.x (development) | :white_check_mark: |
| 0.3.x | :white_check_mark: |

## Reporting a vulnerability

Please report security vulnerabilities privately to the maintainers listed in
`MAINTAINERS.md`.  Do **not** open a public issue.

We aim to acknowledge reports within 5 business days and provide an initial
assessment within 15 business days.  Please do not disclose a vulnerability
publicly until a fix has been released and a reasonable window has elapsed.

## Scope

- Supply-chain integrity of the released wheel/sdist.
- Numerical correctness that changes financial conclusions (reported as
  security-relevant when it can cause material misstatement of risk).

## Out of scope

- GIPS / Basel / regulatory compliance certification (fincore provides
  calculation support only).
- Vulnerabilities in optional data-provider SDKs (report upstream).
