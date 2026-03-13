# Cloudflare Autodeploy Setup

This repository is configured for GitHub-driven Cloudflare Workers deployment, matching the CEL deployment pattern.

## What is configured

- `wrangler.toml`
  - production worker: `hazardpulse`
  - preview worker: `hazardpulse-preview`
  - static assets directory: `dist`
- `.github/workflows/deploy.yml`
  - push to `main` -> production deploy
  - pull request to `main` -> preview deploy
  - manual trigger -> production deploy

## Required GitHub Secrets

In `GitHub -> Settings -> Secrets and variables -> Actions`, add:

- `CLOUDFLARE_API_TOKEN`
- `CLOUDFLARE_ACCOUNT_ID`

The token must have Workers deploy permissions for your account.

## Required build output

The workflow expects a `dist/` directory in repo root.

Before pushing, ensure your site build writes static output there.
If `dist/` is missing, deploy fails intentionally.

## Quick verification

1. Commit and push these files.
2. Confirm `Deploy to Cloudflare Workers` workflow appears in Actions.
3. Run `workflow_dispatch` once to validate production deploy.
4. Open a test PR to validate preview deploy.

## Notes

- If you want a different worker name, update `wrangler.toml`.
- If you want branch-based production (for example `coherence-prime`), update workflow trigger branches.

