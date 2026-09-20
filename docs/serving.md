# Serving

The API, the container, and the Kubernetes deployment — what each one is for,
and which decisions were arguable.

## What is actually served, and why it is batch-first

A fantasy projection is a weekly batch product. The board for week *W* is
computed once, before *W* kicks off, and then it is a fact about that week. It
must not change as Sunday progresses — if it did, every claim in
[`leakage.md`](leakage.md) would collapse, because a number that moves after
kickoff is a number that saw the outcome.

So the primary endpoints read an **artifact**, not a live model. Building this
as a low-latency online scoring service would be architecture theatre: nobody
needs sub-100ms inference for a number that changes once every seven days, and
pretending otherwise would mean building infrastructure whose only purpose is
to look impressive.

`POST /predict` does run the predictor live, on history the caller supplies.
That is a what-if tool, and it is honest about being one. It takes history
rather than a player id deliberately — a request naming only a player would
either be a lookup wearing a prediction's clothes, or the server quietly
reading data the caller never saw.

## The bundle

`nflproj publish` writes a directory with two files:

```
bundle/projections.parquet   the board
bundle/bundle.json           what the board claims about itself
```

The manifest carries the predictor, the season and week, the week's first
kickoff, the SHA-256 of every upstream file the board was built from, and the
predictor's measured metrics at publish time. Three properties matter:

**The version is a content hash.** Rebuilding from the same data with the same
predictor produces the same version string, so *"is production serving what I
think it is"* is answerable by comparison rather than by trust. `created_at` is
excluded from the hash on purpose — otherwise every rebuild would look like a
new model.

**Loading verifies the hash.** A truncated copy, a half-written volume or an
edited parquet makes `load_bundle` raise. This is not decoration: without it the
service would answer with projections nobody generated, and every provenance
field in the response would be a lie that looks exactly like the truth.

**The bundle records when it stops being a prediction.** `valid_from` is the
week's first kickoff. Past that moment the board is a record of what was
claimed, and the API sets `superseded: true` rather than letting a caller
present a settled week as advice. The board stays readable either way — the
track record depends on it.

Publishing a week that has already started is a warning, not an error. The
numbers are uncontaminated (`project_week` only reads settled weeks, so a late
run is bit-identical to a timely one) — they are simply late, and *late* and
*wrong* deserve different words.

## Liveness and readiness are different questions

| Endpoint | Question | False when |
|---|---|---|
| `/healthz` | Is this process alive? | The process is wedged. Restart it. |
| `/readyz` | Should this pod get traffic? | No bundle, or the bundle failed its hash check. |

Wiring both probes to one endpoint is the standard shortcut and it fails in two
opposite directions at once: Kubernetes restarts pods that are merely still
loading, *and* routes traffic to pods holding a corrupt bundle. A missing bundle
is not a hung process, and a restart cannot fix it.

A bundle that fails verification leaves the service permanently not-ready
rather than falling back to anything. In a repo whose entire claim is that you
can check where the numbers came from, serving numbers of unknown origin is
worse than serving none.

## The image is the model version

The bundle is baked into the container image. The tag *is* the model version,
and a rollback is redeploying last week's tag.

The argument: a weekly cadence makes immutability nearly free. There is no model
store to be unavailable, no runtime fetch to fail, and nothing that can drift
between what CI tested and what is serving. The rollout strategy is
`maxUnavailable: 0`, so the previous model keeps serving until the new one has
passed a readiness probe that includes its integrity check.

The cost, stated plainly: **updating the model means redeploying.** For a daily
or hourly model that would be the wrong trade and a mounted volume or a model
registry would win. For a board that changes on Tuesdays it is the boring
choice, and boring is what this repo prefers.

`bundle/` always exists in the build context, so `docker build` works without
publishing first. The resulting image simply comes up not-ready and says why —
which is the correct behaviour for a serving container with nothing to serve,
and it exercises the readiness path.

## What CI actually proves

Writing Kubernetes YAML proves nothing. The `container + kubernetes` job:

1. builds a real bundle from live nflverse data,
2. builds the image and asserts it runs unprivileged,
3. stands up a **kind** cluster and applies the real manifests through the kind
   overlay, which changes exactly one thing — the image reference,
4. waits for the rollout, which only goes green if readiness passes, which only
   passes if the bundle verified,
5. port-forwards the Service and asserts a projection comes back with a
   non-zero score, a 16-character bundle version and a non-empty set of upstream
   hashes,
6. posts to `/predict` and asserts the live path works too,
7. pushes to GHCR — only from `main`, and only after all of the above.

Step 5 is the one that matters. A 200 would pass with an empty board, so the
assertions are on the contents. And provenance is checked at the end of the
whole path — publish → image → pod → Service — because that is the point where
someone would actually consume the number.

## Metrics

Prometheus text format at `/metrics`, scraped via pod annotations.

Requests and latency are labelled by **route template**, never raw path. That is
a deliberate guard with a test behind it: labelling `/projections/2026/3/p1` by
path would mint a new time series per player id, and that is how a metrics store
gets taken down by a service that looks perfectly healthy.

## Not done, and known

- **No standing cluster.** The deployment is proven in CI on kind. A managed
  cluster is a recurring bill and a maintenance surface, and a dead demo link is
  worse than no link. The manifests are cloud-portable, so standing one up is a
  decision rather than a rewrite.
- **The HPA is applied but not exercised.** kind has no metrics-server, so CI
  proves the manifest is valid and nothing more.
- **No authentication, rate limiting or TLS.** This serves public projections.
  Adding auth to something with nothing to protect would be cargo cult; an
  Ingress terminating TLS is the natural place for it if that changes.
- **The served predictor is still a baseline**, not a learned model. The
  serving layer is complete and the model is next. The `Predictor` protocol is
  what makes that a swap rather than a rewrite — which is the same reason the
  backtest harness needs no changes to score an XGBoost model.
