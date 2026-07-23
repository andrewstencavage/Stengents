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
In a second terminal, start the SSH forward and leave it running:

```bash
ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -N \
  -L 127.0.0.1:11434:127.0.0.1:11434 gym
```

Configure the local OpenAI-compatible endpoint in your shell. Do not commit
these settings or any real API token:

```bash
export STENGENTS_MODEL_BASE_URL=http://127.0.0.1:11434
export STENGENTS_MODEL_NAME=llama3.1:8b
export STENGENTS_MODEL_API_KEY=local
```

Run the first fixture:

```bash
stengents run normalize-index
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
PYTHONPATH=src .venv/bin/python -m pytest
```
