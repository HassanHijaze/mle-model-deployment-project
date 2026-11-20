# Operations: promote and roll back a model

The API serves the version of `nyc-taxi-duration-rf` that has the **`champion`** alias. Nothing moves
that alias automatically: the pipeline registers new versions without an alias, and you move it by
hand. `/reload` needs the `X-Admin-Token` header (value: `ADMIN_TOKEN` from `.env`).

## Selection rule (fixed before the results)

Data: January 2025. Every model is trained on Jan 1–20 only, compared on validation (Jan 21–25), and
only the winner is scored on test (Jan 26–31), once, next to the baselines.

Candidates: the RF baseline (`nyc-taxi train`) and every Optuna trial (`nyc-taxi tune`).

1. Best: the lowest validation RMSE.
2. Tied: every candidate within 1% of the best validation RMSE.
3. Winner: the smallest model (`model_mb`) among the tied candidates.

## Promote a new version

1. **Register.** `uv run nyc-taxi all` (or `select` after `train` and `tune`) applies the selection
   rule above, scores the winner once on test and registers it; it prints the version number.
2. **Verify.**

   ```bash
   uv run nyc-taxi verify <version>
   ```

   Loads that version in a fresh process, checks the feature schema and the zone table, predicts a
   few known trips and prints `OK` or the exact problem. Do not promote a version that fails.
3. **Set champion.** In the MLflow UI (http://localhost:5002) → Models → `nyc-taxi-duration-rf` →
   the version → add the alias `champion`. The previous champion loses the alias but stays
   registered.
4. **Reload the API.**

   ```bash
   curl -X POST http://localhost:8000/reload -H "X-Admin-Token: $ADMIN_TOKEN"
   ```

   The API loads the model, zone table and feature schema as one bundle and validates it before
   switching. Requests keep using the old model while it loads.
   - `200`: `current_version` is the new champion.
   - `422 bundle_invalid`: the version failed validation (`problems` lists why).
   - `502 reload_failed`: MLflow or the model could not be loaded.
   - `409 reload_in_progress`: another reload is running.

   For every error, the old model keeps serving (`serving_version`).
5. **Check.** `curl http://localhost:8000/ready` shows the version being served, and every
   `/predict` response includes `model_version`.

## Roll back

1. In the MLflow UI, move the `champion` alias back to the previous version.
2. Reload: `curl -X POST http://localhost:8000/reload -H "X-Admin-Token: $ADMIN_TOKEN"`.
3. Confirm with `curl http://localhost:8000/ready`.

## Probes and logs

- `GET /live`: the process is up (use for liveness / restart decisions).
- `GET /ready`: a validated model is loaded (use for readiness / traffic decisions).
- `GET /health`: model, zone table and the result of the last reload.
- Logs are JSON lines with `request_id` (also returned as the `X-Request-ID` header) and
  `model_version`, so one request can be traced from the response to the logs.
