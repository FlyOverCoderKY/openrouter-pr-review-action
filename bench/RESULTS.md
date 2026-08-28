# Bench results

Recorded runs of the offline recall bench (see [README.md](README.md)).
Numbers are means over the listed runs; lanes are nondeterministic, so
compare means, not single runs. Costs are approximate, from OpenRouter
usage at the listed provider's pricing on the run date.

> Scoring note: results recorded on 2026-08-27 predate the three-way
> adjudication change — their precision column counted every unmatched
> finding as false. Later runs report adjudicated precision plus a separate
> `noise` column, so precision numbers are not directly comparable across
> that boundary (recall numbers are).

## 2026-08-28 — falsification-pass prompt A/B (x-ai/grok-4.6, 5 runs/arm)

The research report's experiment #1: adding an asymmetric falsification pass
to the initial prompt (falsify each draft bug/risk against callers, guards,
tests, and framework guarantees; name searches behind absence claims; drop
only on direct counterevidence — uncertainty stays as a stated proof gap).
First experiment scored under three-way adjudication with the noise column
and both twins.

| arm | recall | bug | diff-stratum | sev-agree | noise (planted) | clean-twin findings/run |
| --- | --- | --- | --- | --- | --- | --- |
| v1.2.2 baseline | 91% | 80% | 90% | 30% | 2% | 5.0 |
| + falsification | **95%** | 80% | **94%** | **42%** | **0%** | **3.8** |

**Adopted**: no downside cell — recall rose (asymmetric guardrail held),
severity calibration improved, noise fell on both fixtures. Two side
lessons: (1) baseline recall at n=5 is 91%, not the 97-100% the earlier n=3
rows suggested — small samples flattered every model above; (2) the clean
twin's findings are defensible minor observations about genuinely imperfect
corners of the "clean" code (unvalidated negatives in apply_cap, loose
exception asserts, under-cap paths untested), NOT fabrications — zero
invented defects in 10 clean runs. As built, the clean twin measures the
chattiness floor rather than hallucination; harden the clean head or
adjudicate its recurring observations to sharpen the question.

**Cross-model validation (z-ai/glm-5.3-flash, 5 runs/arm)** — the shared
prompt must not regress other lanes: GLM recall 98%→98% (bug 100%→100%, no
stratum regressed), clean-twin volume 6.6→6.2 findings/run, noise 2%→4%
(one finding in 25). Neutral-to-slightly-positive; the over-drop failure
mode did not appear on a second model family.

**Cost of the pass** (fixture-scale): grok +0.8 tool rounds (2.6→3.4) and
+22% wall time on planted (+5% on clean); GLM +0.2 rounds and +7-14% time.
Far from the 50-turn budget here — but this fixture is tiny, and behavior
against a dense real PR near the turn/observation budget is an open proof
gap (a real-PR replay fixture is the planned check). If the budget does
exhaust, tools are withdrawn and a schema finish still returns the findings
gathered so far.

## 2026-08-27 — planted-mini, six models

Fixture: `bench/fixtures/planted-mini` (11 labels: 2 bug / 4 risk / 5 nit),
v1.2.2 exhaustive prompt (PR #7 branch), production effort (empty), 50 tool
turns, 3 runs per model.

| model | provider | recall | bug recall | precision | ~cost/review | mean time | lanes ok |
| --- | --- | --- | --- | --- | --- | --- | --- |
| moonshotai/kimi-k3 | Fireworks | **100%** | **100%** | 100% | ~$0.14 | ~95s | 2/3 ¹ |
| x-ai/grok-4.6 | xAI | 97% | 83% | 100% | $0.049 | 68s | 3/3 |
| z-ai/glm-5.3-flash | Z.ai | 97% | 83% | 97% | **$0.002** | 149s | 3/3 |
| qwen/qwen3.8-27b | AkashML (bf16) | 91% | 67% | 94% | $0.034 | 197s | 3/3 |
| google/gemini-3.7-flash | Google | 82% | 50% | 100% | $0.020 | **42s** | 3/3 |
| nvidia/nemotron-3.5-lightning:free | free tier | 55% | 25% | 100% | $0 | 489s | 2/3 ² |

¹ Third lane hit an upstream shared-pool rate limit (Fireworks' Kimi
capacity across OpenRouter users), not a harness failure.
² One timeout on the free route.

**The tier discriminator** is label B1 (a stale dollar value that only
falls to checking the figure against its cited source — the same class
that decided the live side-by-side reviews): kimi 2/2, grok 2/3, glm 2/3,
qwen 1/3, gemini 0/3, nemotron 0/3. The leaderboard is essentially a
ranking of who does that verification work.

**Roster take:** glm-5.3-flash delivers grok-tier recall at ~1/25th the
cost (volume lane); kimi-k3 is the only model that has never missed a
label (premium depth lane); grok-4.6 remains the incumbent benchmark.
qwen3.8-27b is dominated by glm on every axis; gemini is fast and precise
but misses the verification class; the free tier is a universality proof,
not a reviewer.

### Prompt A/B (same fixture, x-ai/grok-4.6, 3 runs each)

The v1.2.2 exhaustive prompt vs the v1.2.1 prompt, the change PR #7 ships
(measured before the fixture's custom_instructions de-leak, on the
original 10 labels):

| prompt | recall | bug recall | notes |
| --- | --- | --- | --- |
| v1.2.2 (recall port) | 100% (30/30) | 6/6 | caught B1 in every run |
| v1.2.1 | 93% (28/30) | 4/6 | missed B1 in 2 of 3 runs |

### GLM-5.3-flash provider shootout (same fixture, 3 runs each)

GLM's wall time is its own 7-9K-token completions at ~55 tps, not the
host. Pinning uses `--provider` (fallbacks disabled); check a provider's
`structured_outputs` support before pinning — the schema-enforced
finalization requires it.

| provider | lanes ok | mean time | notes |
| --- | --- | --- | --- |
| Z.ai (default routing) | 3/3 | 149s | 50%-off pricing; recommended |
| Together | 3/3 | 134s | ~10% faster at 2× price |
| Novita | 1/3 | 176s | capacity 404s when pinned |
| BaseTen | 0/3 | — | no structured outputs |
