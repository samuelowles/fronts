# Security

`fronts` executes Python that a language model wrote. That is the point of the
architecture, and it is also the entire security story. A repository that does
this and does not say so plainly should not be trusted.

## The threat model

The synthesised world model is code produced by an LLM from your platform
rules and your posting history. No human reviews it before it runs. It is not
signed. If your provider is compromised, if your rules file contains a prompt
injection, or if the model hallucinates something destructive, the output is a
Python module that `fronts` will load and call thousands of times inside a
search loop.

So the question is not whether the sandbox is good. The question is what
happens when it fails.

## What the sandbox does

`src/fronts/cwm/sandbox.py` refuses, at compile time via AST inspection:

- every `import` and `from ... import` statement, without exception;
- `open`, `exec`, `eval`, `compile`;
- `__subclasses__`, `__globals__`, `__code__`, and dunder attribute access
  generally.

It then executes with a restricted `__builtins__` and a namespace of pre-bound
values: `sqrt`, `log`, `Random`, `dataclass`, `deque` and so on. Never
modules.

That last distinction matters most, and it was learned the hard way.

## The bug that shaped this file

The first version of the sandbox allowlisted modules: `math`, `random`,
`json`, `dataclasses`, `typing`, `collections`, `re`, `datetime`,
`statistics`. An independent review found nine working escapes and
demonstrated each by execution. The shortest:

```python
import random
random._os.system("...")     # random._os is the real os module
```

Others reached the filesystem through `typing.sys.modules['os']`,
`dataclasses.builtins.open`, `collections._sys`, `json.codecs`, and
`statistics.random._os`.

None of them used a forbidden attribute name, so a longer denylist would not
have helped. The failure was the policy, not the enforcement: an allowlist of
modules is not a boundary, because module objects form a reachable graph.

The fix was to remove the graph. There are no module objects in the namespace,
so none of these attacks has a first step. The load-bearing test is not the
nine regressions. It is this one:

```python
assert not any(isinstance(v, ModuleType) for v in vars(namespace).values())
```

That covers the bypasses nobody has thought of yet, which the regression tests
by construction cannot.

## What the sandbox does not do

It is not a security boundary against a determined adversary. It raises the
cost of an accident and blocks the known escape classes. It runs in your
process, with your file descriptors, under your user.

CPython cannot forcibly kill a thread, so `call_with_timeout` protects the
caller's control flow, not the process: a synthesised module that spins will
leak a thread even though the timeout fires correctly.

Know what the shipped CLI wraps and what it does not. The per-call guard
(`Sandbox.guarded`) costs one worker thread per call, and the planning loop
makes on the order of 10^5 model calls per plan, so the loaded world model
runs unguarded in-process. A model whose method body loops forever will hang
`fronts plan` until you kill it. The persisted inference sampler is guarded,
since one call per determinization is cheap, and the AST gate above runs on
everything before it executes. If a hang from hostile source is in your threat
model, use a process boundary rather than threading the inner loop.

If you run synthesis output you have any reason to distrust (a shared rules
file, an untrusted provider, a multi-tenant deployment), use
`SubprocessSandbox`, which executes in a separate process and can actually
kill it, or put the whole thing in a container. The per-call IPC cost makes
`SubprocessSandbox` unsuitable for the inner MCTS loop. That trade-off is
yours to make, and it is the honest price of a real boundary.

## Operational notes

- API keys are read from the environment (`ANTHROPIC_API_KEY`,
  `OPENAI_API_KEY`, `COMPOSIO_API_KEY`) and are never written to trajectory
  files, fixtures, or logs.
- Recorded fixtures under `fixtures/` contain prompts and responses. If you
  re-record against a live provider, check before committing that your rules
  text does not embed anything private. The whole prompt is stored verbatim.
- `fronts publish` is dry-run by default, and `--live` is required for any
  real call, because the failure mode of a planner bug is public.
- Nothing here bypasses platform rate limits, evades detection, fabricates
  engagement, or scrapes behind a login. The legality engine makes several of
  those structurally unrepresentable.

## Reporting a vulnerability

Open a GitHub issue for anything already public, such as a sandbox escape you
can demonstrate. Escapes are more useful in the open, and a working one is a
contribution rather than an embarrassment.

For anything you believe should be handled quietly, use GitHub's private
vulnerability reporting on this repository rather than a public issue.

If you find a sandbox escape, a proof-of-concept in the style of
`test_sandbox_refuses_known_module_graph_escapes` is the most useful possible
bug report, and it will be merged as a regression test with credit.
