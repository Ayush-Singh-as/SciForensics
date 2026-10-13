# Phase A demo deployment

The supported demo is one CPU API process and a Next.js frontend. Jobs and uploads
are temporary; restarting the API clears them. This is sufficient for a demo, and
does not provide Stage C's durable multi-worker queue.

## Local

```bash
pip install -e '.[demo,dev]'
python -m playwright install chromium
sciforensics serve --config configs/demo.yaml
```

In another terminal:

```bash
cd web
npm ci
npm run build
npm start
```

Open **http://localhost:3000**. Run any of the three gallery examples, or upload
images. `configs/demo.yaml` uses CPU, caps the longest analysis dimension at 1280,
and uses the documented CMFD settings for the planted clone. Input coordinates in
reports still refer to original pixels. Both the gallery and checkpoint work
without downloads after installation.

The Python default suite excludes `learned` tests: those need separate neural
matcher weights and network access. Run them explicitly with `pytest -m learned`
when working on Stage B. `pytest -q`, `ruff check .`, and `mypy` are the Phase A
checks. For browser acceptance against running servers:

```bash
python scripts/smoke_demo.py
# Optional: regenerate the README recording from the verified flow
python scripts/smoke_demo.py --record docs/assets/demo.gif
```

PyMuPDF is required only for the browser test's PDF text extraction:
`pip install PyMuPDF`. The script checks examples, uploads, synchronized zoom and
pan, SVG correspondences, overlays, local threshold re-scoring, WebSocket progress,
PDF download/provenance, negative controls, and CMFD.

## Docker

```bash
SCIFORENSICS_BUILD_REVISION=$(git rev-parse HEAD) docker compose up --build
```

Open **http://localhost:3000**; the API is at **http://localhost:8000/docs**. The
API image contains the checksum-verified `models/weights.pth` and packaged example
assets, and uses WeasyPrint's installed native libraries for PDFs. An unpublished
GitHub release URL cannot break the container's first analysis.

## Fly.io backend + Vercel frontend

1. Authenticate to Fly.io and create an app from the repository root:

   ```bash
   fly launch --no-deploy --dockerfile docker/api.Dockerfile --region sin
   ```

   Keep the supplied `fly.toml` CPU/memory settings and `docker/api.Dockerfile`.
   Choose a unique app name. No database or volume is required.
   The supplied region is `sin` (Singapore). Check `fly platform regions` before
   choosing another region. A successful launch writes `app = "<your-app>"` to
   `fly.toml`; complete launch before running `deploy` or `scale`.

2. In Vercel, import this repository with **Root Directory = `web`**. Add the
   environment variable `SCIFORENSICS_API=https://<your-fly-app>.fly.dev` for the
   Production environment before building. This URL is baked into the browser
   client; changing it requires a Vercel rebuild. Deploy to obtain its production
   origin, for example `https://<your-project>.vercel.app`.

3. Set the exact browser origin on Fly.io (no trailing slash) and deploy:

   ```bash
   fly secrets set 'SCIFORENSICS_API__CORS_ORIGINS=["https://<your-project>.vercel.app"]'
   fly deploy --build-arg BUILD_REVISION="$(git rev-parse HEAD)"
   fly scale count 1
   ```

   Keep one Machine: the job store is process-local, and multiple Machines would
   route polling and assets to different stores. Fly autostop can clear old jobs;
   use `min_machines_running = 1` for a live presentation if cold-start latency is
   undesirable. Include every intended preview origin explicitly when testing
   Vercel previews. Uploads and reports go directly to Fly rather than through
   Vercel's request body limits. WebSocket progress uses the same backend URL;
   polling remains available if a network blocks WebSockets.

4. Verify `https://<your-fly-app>.fly.dev/healthz`, then run the gallery from the
   Vercel URL. A successful health check alone does not verify model inference or
   PDF rendering. Run the browser acceptance test with `DEMO_WEB_URL` set to the
   deployed frontend URL.

Hosting accounts and CLI credentials are required to execute these deployment
steps. No public URL has been created by the repository changes alone.

References: [Fly configuration](https://fly.io/docs/reference/configuration/),
[Fly secrets](https://fly.io/docs/apps/secrets/),
[Vercel environment variables](https://vercel.com/docs/environment-variables).

## Checkpoint release

The checkpoint remains tracked so a clone can build the demo immediately. Git
history is unchanged. To distribute it separately, run the **Publish checkpoint**
GitHub Actions workflow with a release tag. The workflow verifies its pinned SHA-256
before uploading. Once the release exists, set the actual HTTPS asset URL as
`SCIFORENSICS_WEIGHTS_URL` or pin it in `weights.RELEASE_URL`; missing or corrupted
artifacts must fail checksum verification. No release URL is invented in advance.
