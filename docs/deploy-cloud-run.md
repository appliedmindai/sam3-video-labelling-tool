# Deploying to Google Cloud Run

Deployment is now driven by a single script — see **[DEPLOY.md](../DEPLOY.md)**
at the repo root (humans) and **[deploy/README.md](../deploy/README.md)**
(coding-agent quick-reference).

```bash
PROJECT=your-project-id ./deploy/deploy.sh up
```

The manual step-by-step gcloud walkthrough that used to live here was
folded into DEPLOY.md when the script was introduced (2026-06-11); the
git history of this file preserves the original.
