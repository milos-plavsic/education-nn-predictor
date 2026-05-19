# Security Policy

## Supported Versions

Security fixes are applied to the most recent minor release on the `main`
branch. Older releases receive fixes only for issues rated High or Critical.

| Version | Status         |
| ------- | -------------- |
| `main`  | Actively fixed |
| `<n-1>` | Critical only  |
| Older   | Unsupported    |

## Reporting a Vulnerability

Please **do not** open a public issue for security reports. Instead:

1. Email a detailed report to the maintainer listed in `CODEOWNERS` (or
   `milos.plavsic@googlemail.com` if no CODEOWNERS file is present).
2. Include reproduction steps, affected versions, and suggested mitigations
   if you have them.
3. You will receive an acknowledgement within **72 hours**.

Once a fix is available we will:

- Coordinate a disclosure timeline with you.
- Credit you in the release notes (unless you prefer to remain anonymous).
- Publish a GitHub Security Advisory and request a CVE where applicable.

## Scope

In scope:

- Code in this repository (application, infrastructure, CI workflows).
- Default configurations shipped with the project.

Out of scope:

- Vulnerabilities in third-party dependencies that have an upstream fix
  pending — please report those upstream.
- Issues that require physical access or social engineering.
- Denial-of-service from unbounded request volume against a single instance
  with no rate limiting configured.

## Hardening Already in Place

- Container images run as a non-root user.
- HTTP responses include `X-Content-Type-Options`, `X-Frame-Options`,
  `Strict-Transport-Security`, `Referrer-Policy`, and a default-deny CSP.
- Inbound requests are validated by Pydantic; rate limiting and per-request
  timeouts are enabled.
- Dependencies are pinned and scanned on every push (Bandit, Safety, Trivy).
- Dependabot opens PRs for security advisories on a daily schedule.

## Cryptographic Material

This project does not store secrets in source. Configuration is loaded from
environment variables via `pydantic-settings`. Use a secret manager
(Kubernetes Secrets, AWS Secrets Manager, Vault) in production deployments.
