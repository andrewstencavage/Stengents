# ADK local-model provider adapter: findings for issue #2

## Answer

Use one Python adapter that creates ADK's `LiteLlm` model connector for an
OpenAI-compatible endpoint:

```python
from google.adk.models.lite_llm import LiteLlm


def local_model(*, base_url: str, model: str, api_key: str = "local") -> LiteLlm:
    return LiteLlm(
        model=f"openai/{model}",
        api_base=f"{base_url.rstrip('/')}/v1",
        api_key=api_key,
    )
```

Configure `base_url`, `model`, and any token through the process environment
(or a local, ignored `.env` file), not Python source. The ADK Python quickstart
uses a project `.env` for API credentials; repository policy should ensure any
such local file is not committed. [Python quickstart](https://adk.dev/get-started/python/)

This is the MVP recommendation: select the tunnelled gym's Ollama or LM Studio
server by configuration only, provided it exposes an OpenAI-compatible endpoint.
ADK explicitly documents this exact `openai/<model>` plus `OPENAI_API_BASE`
`.../v1` route for Ollama. ADK does not name or certify LM Studio in its model
documentation, so its use is an interoperability assumption to verify against
the gym endpoint—not a claim of direct ADK support. [ADK Ollama: OpenAI provider](https://adk.dev/agents/models/ollama/#use-openai-provider)

## Supported choices and minimum dependencies

| Choice | ADK status | Configuration | MVP fit |
| --- | --- | --- | --- |
| OpenAI-compatible adapter | ADK's documented Ollama route via `LiteLlm` | `model="openai/<endpoint-model>"`; base URL ends in `/v1`; API key may be a local placeholder if the server requires one | **Choose this.** One adapter shape for the tunnelled service. |
| Native Ollama adapter | Documented via `LiteLlm` | `model="ollama_chat/<model>"`; `OLLAMA_API_BASE` points to the server | Optional fallback only. It is Ollama-specific. |

Install ADK and LiteLLM:

```bash
pip install google-adk litellm
```

ADK's installation guide specifies `google-adk`; its LiteLLM connector guide
specifies installing `litellm`. The model overview identifies `LiteLlm` as a
model connector passed to an `LlmAgent`, including for Ollama-hosted models.
[Installation](https://adk.dev/get-started/installation/) · [LiteLLM connector](https://adk.dev/agents/models/litellm/) · [Model connectors](https://adk.dev/agents/models/)

## Boundaries and validation

- The documented native Ollama form is `ollama_chat`, not `ollama`; ADK warns
  that `ollama` can cause tool-call loops and loss of prior context. [ADK Ollama](https://adk.dev/agents/models/ollama/)
- For coding-agent tools, the serving endpoint must support and enable
  OpenAI-compatible function/tool calling. ADK calls this out for compatible
  self-hosted endpoints. [ADK vLLM](https://adk.dev/agents/models/vllm/)
- ADK's self-hosted endpoint example passes `api_base`, `api_key`, and headers
  directly to `LiteLlm`, so the adapter can avoid global provider-specific
  environment variables. [ADK vLLM integration example](https://adk.dev/agents/models/vllm/#integration-example)

## Minimal integration shape

```python
import os

from google.adk.agents import LlmAgent

root_agent = LlmAgent(
    name="coding_agent",
    model=local_model(
        base_url=os.environ["STENGENTS_MODEL_BASE_URL"],
        model=os.environ["STENGENTS_MODEL_NAME"],
        api_key=os.environ.get("STENGENTS_MODEL_API_KEY", "local"),
    ),
    instruction="Work on the coding fixture.",
)
```

Before adding coding tools, run a smoke prompt through the tunnel and a single
function-call fixture. This establishes that the configured model name, `/v1`
path, and tool-calling behavior match the local server.
