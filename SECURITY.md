# Security Policy

## Overview

Thank you for helping keep the SIM-ONE Framework and its users safe.
This policy explains how to report vulnerabilities, what is in scope, and how we coordinate disclosure.

## Supported Versions

We provide security fixes for:
- The latest release on the main branch
- The most recent minor release when there is an active LTS tag in this repository

Older releases may receive fixes case by case if impact is critical and the patch is low risk.

## Reporting a Vulnerability

Please email **security@gorombo.com** with the subject **[SECURITY] Vulnerability Report**.

Include:
- A clear description of the issue and impact
- A minimal proof of concept or reproduction steps
- Affected version or commit hash and environment details
- Any logs, stack traces, screenshots that help reproduce
- Contact information for coordinated follow up

Do not open public GitHub issues for suspected vulnerabilities.

If the issue involves a third-party dependency, report it to us and the upstream project when possible.

## Our Response Process

- **Acknowledgment** within 3 business days
- **Triage** and severity rating (CVSS 3.1 guidance) within 7 business days
- **Fix plan** provided after triage with an estimated timeline
- **Advisory** will be published when a fix or mitigation is available

We will keep you informed during triage and remediation and invite you to validate the fix where appropriate.

## Coordinated Disclosure

Default embargo period: **90 days** from acknowledgment.
We may shorten or extend this period depending on exploitability, user impact, or availability of mitigations.

We credit reporters in release notes and advisories unless you request anonymity.

## Scope

In scope:
- Vulnerabilities in SIM-ONE Framework source code and official packages
- Security defects in default configuration and documented deployment paths
- Supply chain risks that affect how the framework is built or distributed

Out of scope:
- Social engineering, phishing, and physical attacks
- Denial of service through excessive traffic without a specific bug
- Self-XSS that requires the victim to paste code into their own console
- Vulnerabilities in third-party services or dependencies that are not exploitable through our usage
- Issues on non-production demo sites that do not affect the software itself

If you are unsure about scope, send the report anyway. We will triage it.

## Safe Harbor

We will not pursue or support legal action against researchers who:
- Act in good faith and comply with this policy
- Avoid privacy violations and avoid destruction of data
- Do not disrupt production systems
- Do not attempt to access data that does not belong to you
- Give us a reasonable time to remediate before public disclosure

## Security Updates and Advisories

Security fixes are announced in:
- GitHub Releases
- Repository CHANGELOG
- GitHub Security Advisories (GHSA) when applicable

For critical fixes we will provide upgrade guidance and mitigations.

## Dependencies and Supply Chain

We monitor vulnerabilities in direct dependencies and build tooling.
If you discover a vulnerability in our build or release process, report it through the email above.

## Contact

security@gorombo.com

For commercial customers with support contracts, use your support channel in addition to the email above.
