# Security Policy

## Supported versions

| Version | Supported |
| --- | --- |
| 0.4.x (development) | :white_check_mark: |
| 0.3.x | :white_check_mark: |

## Reporting a vulnerability

**Primary channel: GitHub Private Vulnerability Reporting** — open
<https://github.com/cloudQuant/fincore/security/advisories/new> (the
"Report a vulnerability" button on the repository's **Security** tab).

> ⚠️ **Pending enablement confirmation:** as of the 2026-08-23 governance
> baseline, Private Vulnerability Reporting is not yet enabled on this
> repository; the administrator has committed to turning it on. If the button
> is not visible yet, it means enablement is still in progress — do **not**
> open a public issue and do not include exploit details anywhere public.

Do **not** open a public issue for security reports.

We aim to acknowledge reports within 5 business days and provide an initial
assessment within 15 business days. Please do not disclose a vulnerability
publicly until a fix has been released and a reasonable window has elapsed.
Maintainer responsibilities for security response are listed in
[MAINTAINERS.md](MAINTAINERS.md).

## Scope

- Supply-chain integrity of the released wheel/sdist.
- Numerical correctness that changes financial conclusions (reported as
  security-relevant when it can cause material misstatement of risk).

## Out of scope

- GIPS / Basel / regulatory compliance certification (fincore provides
  calculation support only).
- Vulnerabilities in optional data-provider SDKs (report upstream).
