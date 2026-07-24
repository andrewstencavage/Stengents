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
Configure this shell once so future runs need no environment setup. Do not add a
real API token to your shell profile:

```bash
printf '\n# Stengents local model\nexport STENGENTS_MODEL_BASE_URL=http://127.0.0.1:11434\nexport STENGENTS_MODEL_NAME=llama3.1:8b\nexport STENGENTS_MODEL_API_KEY=local\nalias stengents-gym-tunnel='\''ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -N -L 127.0.0.1:11434:127.0.0.1:11434 gym'\''\n' >> ~/.zshrc && source ~/.zshrc
```

In a second terminal, start the SSH forward and leave it running:

```bash
stengents-gym-tunnel
```

If the tunnel command reports that port `11434` is already in use, a tunnel is
usually already running. Leave it alone and run the fixture; only start a new
tunnel after the existing listener stops.

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
