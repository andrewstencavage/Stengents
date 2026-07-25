# Stengents

Stengents is a reproducible harness for a Python ADK coding agent. The first
vertical slice repairs a small deterministic Python fixture and writes a local
JSON run record.

## Run the vertical slice locally

Create and activate a virtual environment, then install the project and its
fixture verifier:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e . pytest
```

The development-time model is served on `gym` and remains loopback-only there.
Load the connection into your shell with the provided script — it sets only the
stable base URL and API key, writes nothing to your shell profile, and never
pins a model. Run it without `eval` first to inspect exactly what it sets:

```bash
eval "$(scripts/stengents-env.sh)"
```

In a second terminal, start the SSH forward and leave it running. One tunnel
serves every model, so you never need a second one:

```bash
scripts/gym-tunnel
```

If the tunnel command reports that port `11434` is already in use, a tunnel is
usually already running. Leave it alone and run the fixture; only start a new
tunnel after the existing listener stops.

Run the first fixture, choosing the model per run with `--model` (falling back
to `STENGENTS_MODEL_NAME` if you prefer to set it in the environment):

```bash
stengents run normalize-index --model llama3.1:8b
```

The endpoint selects models by name per request, so running several models is
just a matter of changing the flag — no extra tunnels or endpoints:

```bash
for model in llama3.1:8b qwen2.5:7b-8k; do
  stengents run normalize-index --model "$model"
done
```

The command first verifies the tunnel, selected model, and required tool-call
support. On success it prints a credential-free startup JSON line with the
exact `record_path`, then runs in an ephemeral fixture copy. The final JSON
run record is stored at:

```text
.stengents/runs/<run-id>.json
```

To inspect generated records:

```bash
find .stengents/runs -name '*.json' -print
```

If preflight fails, no run record is created. Ensure the SSH tunnel is running
and retry; a cold local model may need one retry before its bounded tool-call
preflight completes.

## Verify the harness

```bash
.venv/bin/python -m pytest
```
