Drive a Backtest with a Self-Hosted Open Model (vLLM)
====================================================

``agentictrading.integrations.vllm`` connects an open-weight model served by
`vLLM <https://docs.vllm.ai>`_ on an NVIDIA GPU to the Lab's Agent-Environment
Protocol. It is an ordinary external agent: the Lab still owns the market data,
the simulation clock, execution and every risk limit; the adapter only turns one
step into one prompt and one reply into one decision.

.. code-block:: text

   ATL step (observation + constraints + deadline_at)
     -> deterministic snapshot serializer (allow-listed fields, no future bars)
     -> POST {VLLM_BASE_URL}/chat/completions (JSON-schema structured output)
     -> strict parser (JSON-or-nothing, orders-v1 subset)
     -> Decision (or an explicit empty-orders HOLD naming the failure)
     -> AgentRunner submits it; ATL validates, executes and records it

.. contents::
   :local:
   :depth: 2


What already existed, and what was added
----------------------------------------

- **Reused unchanged:** the protocol Run API (``/api/v1/runs``), ``ATLClient``,
  ``AgentRunner``, the ``orders-v1`` decision schema (``Decision``/``Order``),
  the server-side checks (allowed symbols, ``max_orders``, position cap, share
  ceiling), the 60 s decision window with auto-hold, and the step pin that makes
  a late decision ``decision_deadline_exceeded`` instead of applying it to the
  next step.
- **Not usable for this:** the Lab's hosted model providers. Their HTTP client
  refuses non-public addresses (``infrastructure/llm/adapters/safe_http.py``),
  so a private GPU server cannot be registered there, and should not be exposed
  publicly to make it fit.
- **Added:** an optional ``on_step(step)`` hook on ``AgentRunner`` (fires just
  before ``decide``; agents without it are unaffected), because ``decide`` only
  receives the observation, while a deadline-aware agent needs
  ``step.deadline_at`` and ``step.constraints``. And the adapter itself, standard
  library only, like the rest of the SDK.


Pinned Stage 1 configuration
----------------------------

======================  ====================================================
Model                   ``Qwen/Qwen2.5-3B-Instruct``
Revision                ``aa8e72537993ba99e69dfaafa59ed015b17504d1``
                        (weights and tokenizer)
License                 Qwen Research License (research / non-commercial;
                        read it before any other use)
Weights                 ~6.2 GB (bfloat16 safetensors)
Server                  ``vllm==0.31.0`` (``requirements-vllm-server.txt``)
Precision               bfloat16; ``VLLM_DTYPE=float16`` on pre-Ampere GPUs
Decoding                temperature 0, top_p 1, seed 0, ``--generation-config
                        vllm`` (the model's own sampling defaults are ignored)
Max output tokens       256
Request timeout         30 s per attempt, and never past ``deadline_at`` - 5 s
Retries                 at most 1, only for connection errors / HTTP 429 / 5xx
Prompt / parser         ``atl-vllm-prompt.1`` / ``vllm-orders-v1-strict.1``
======================  ====================================================

``VLLMConfig.config_hash()`` covers every row above that shapes a request
(plus the system prompt and the output schema), and is stamped on each
decision's ``trace`` and on the agent version. Equal hashes mean an identical
request; GPU kernels are not bit-identical across hardware or batch shapes, so
they do not promise an identical reply. Each request and response is recorded
by SHA-256, never by content, and no record contains the API key.


Serve the model (GPU host)
--------------------------

.. code-block:: bash

   python3 -m venv .venv-vllm && . .venv-vllm/bin/activate
   pip install -r dashboard/examples/vllm_agent/requirements-vllm-server.txt
   pip freeze > vllm-server-freeze.txt          # the resolved lock for this host
   export VLLM_API_KEY="$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')"
   dashboard/examples/vllm_agent/serve_vllm.sh  # binds 127.0.0.1:8000

Health and identity (``/health`` and ``/metrics`` are unauthenticated;
everything under ``/v1`` needs the key):

.. code-block:: bash

   curl -sf http://127.0.0.1:8000/health && echo healthy
   curl -s -H "Authorization: Bearer $VLLM_API_KEY" http://127.0.0.1:8000/v1/models

Example request:

.. code-block:: bash

   curl -s http://127.0.0.1:8000/v1/chat/completions \
     -H "Authorization: Bearer $VLLM_API_KEY" -H "Content-Type: application/json" \
     -d '{"model":"Qwen/Qwen2.5-3B-Instruct","temperature":0,"max_completion_tokens":32,
          "messages":[{"role":"user","content":"Reply with {\"ok\": true}"}]}'

Shutdown: Ctrl-C in the server's terminal, or ``kill <pid>`` (SIGINT/SIGTERM
release the GPU). Check with ``nvidia-smi`` that no ``vllm`` process remains.

Diagnostics:

- ``serve_vllm.sh`` exits 3 without ``vllm``, 4 without ``nvidia-smi``, and 2 when
  asked to bind a non-loopback address without ``VLLM_API_KEY``.
- CUDA out of memory: lower ``VLLM_MAX_MODEL_LEN`` or
  ``VLLM_GPU_MEMORY_UTILIZATION``.
- ``bfloat16 is not supported``: set ``VLLM_DTYPE=float16``.
- 401 from ``/v1``: the client's ``VLLM_API_KEY`` differs from the server's.


Connect securely
----------------

``127.0.0.1`` always means *the machine the code runs on*. When the Lab client
runs on your laptop and vLLM on a GPU box, keep vLLM on loopback and forward
the port over SSH:

.. code-block:: bash

   ssh -N -L 8000:127.0.0.1:8000 user@gpu-host     # laptop: localhost:8000 -> GPU box

Then ``VLLM_BASE_URL=http://127.0.0.1:8000/v1`` on the laptop reaches the GPU
server through the tunnel. A private network (VPN, VPC) with the API key set
works too. Never expose the server on a public address: the key guards ``/v1``
only, and an inference server is a free GPU for whoever finds it.


Run a backtest with it
----------------------

.. code-block:: python

   from agentictrading import AgentRunner, ATLClient
   from agentictrading.integrations.vllm import VLLMChatClient, VLLMConfig, VLLMTradingAgent, summarize_records

   agent = VLLMTradingAgent(VLLMChatClient(VLLMConfig.from_env()))
   result = AgentRunner(ATLClient(atl_url, agent_api_key), agent).run_backtest(
       agent_version_id, environment_id="us-equity-hourly-v1",
       start_date="2026-03-02", end_date="2026-03-03", symbols=["AAPL"])
   print(summarize_records(agent.records))

Client environment: ``VLLM_BASE_URL`` (default ``http://127.0.0.1:8000/v1``),
``VLLM_MODEL_ID``, ``VLLM_MODEL_REVISION``, ``VLLM_API_KEY``,
``VLLM_REQUEST_TIMEOUT`` (seconds, must be below 60), ``VLLM_MAX_OUTPUT_TOKENS``.

Smoke tests (write their reports outside the repository):

- ``vllm_inference_smoke.py --out DIR [--launch]``: fixed synthetic steps,
  server identity, startup time, tokens, latency, TTFT (from vLLM's
  ``/metrics``) and peak GPU memory (``nvidia-smi``; absent means not measured).
- ``vllm_backtest_smoke.py --atl-base-url URL --out DIR [--mock]``: a HOLD
  baseline and the vLLM agent over the same fixed window, with every decision
  record saved. A protocol backtest loads Alpaca bars, so the ATL server needs
  Alpaca credentials. This is a functionality check: its P&L is not a result.


Decision rules
--------------

The model must answer with ``{"orders": [...], "confidence": ..., "rationale":
...}``, where each order is exactly ``{"symbol", "side", "quantity"}`` (whole
shares, market order). The parser refuses the **whole** reply, and the step
becomes an empty-orders HOLD carrying the reason in ``trace``, when:

- the reply is empty, not a single JSON object, has duplicate keys, NaN, or
  extra fields (``empty_response``, ``malformed_json``, ``invalid_schema``);
- a side is not ``buy``/``sell``; a symbol is not exactly in
  ``allowed_symbols`` (no case or whitespace repair); a quantity is not a
  positive integer (``2.0`` and ``"2"`` are refused); a symbol appears twice; or
  there are more than ``max_orders`` orders;
- a sell exceeds the shares held while ``allow_short`` is false;
- confidence is outside [0, 1].

Position weight, cash and per-order size are deliberately **not** checked
here: ATL enforces them and reports them per order. Inference failures
(deadline spent, timeout, connection or HTTP error, unreadable body) also
become a HOLD, so no failed inference can crash the run or place a trade.


Swapping in a fine-tuned adapter (Stage 2)
------------------------------------------

The adapter has no model-specific logic; the model is only ``VLLM_MODEL_ID``.
To compare the base model with the same model plus a LoRA adapter, serve both
from one vLLM process and select by name:

.. code-block:: bash

   VLLM_SERVE_EXTRA_ARGS="--enable-lora --max-lora-rank 64 --lora-modules atl-lora=/path/to/adapter" \
     dashboard/examples/vllm_agent/serve_vllm.sh
   VLLM_MODEL_ID=atl-lora ...      # the fine-tuned arm; Qwen/Qwen2.5-3B-Instruct is the base arm

Snapshot serializer, prompt, schema, parser, ATL interface and metrics stay
identical between the two arms; ``config_hash`` differs only by the model id.
