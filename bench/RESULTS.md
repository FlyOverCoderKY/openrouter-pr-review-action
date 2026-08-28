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

## 2026-08-28 — diversity attribution, corrected: pre-judge diversity is real; the judge loses it

Measured on the hardened fixture. Union arms are computed offline from
saved lanes; the judged row runs the REAL production judge
(google/gemini-3.1-flash-lite) over the same lane pairs. Statistics
corrected after this entry's own self-review: same-model union rates below
are in-sample subset coverage of 5 saved runs (biased upward); the
independence-model expectation is shown beside them.

| config | recall | B1 (measured) | B1 (independence model) | pooled/judged findings | marginal cost basis |
| --- | --- | --- | --- | --- | --- |
| single grok lane | 94% | 40% (2/5) | — | 9.6 | 1.00× ($0.049/review) |
| single glm-5.3-flash lane | 95% | 80% (4/5) | — | ~10; clean 6.8/run | 0.04× ($0.002/review) |
| + source-of-truth profile (grok) | 97% | 60% (3/5) | — | 10.0 | ≈1× + ~15% tokens |
| grok n=2 same-model union (pre-judge) | 98% | 70% in-sample | 64% | 19.2 | 2.00× |
| grok n=3 same-model union (pre-judge) | 99% | 90% in-sample | 78% | 28.8 | 3.00× |
| grok+glm union (pre-judge) | 98% | 88% (22/25) | 88% | 20.8; clean 11.6 | 1.04× |
| **grok+glm THROUGH THE JUDGE (n=5)** | **86%** | **20% (1/5)** | — | 20.8 → 9.8 | 1.04× + judge |

Findings, corrected:

1. Most of the residual B1 miss is run-to-run **variance**, not capability
   (independent 3-lane expectation 78% vs 40% single).
2. The grok+glm union's 88% B1 equals the independence prediction
   1−(0.6×0.2) exactly — the gain comes from GLM's higher marginal B1 rate
   (80% solo) at ~1/25th the price of an extra grok lane, NOT from any
   demonstrated complementary miss profile.
3. **The production judge is a recall bottleneck.** Two-model runs always
   engage the judge, and measured judged output loses the entire union
   advantage and more: 86% recall (below the 94% single lane), B1 1/5,
   pooled findings halved. The judge prompt asks for merge/de-dupe with no
   requirement to retain every distinct input issue, and flash-lite at
   minimal effort deletes aggressively — exactly the "validator that
   silently deletes findings" failure the research report warned against.
4. Personas were NOT measured (they remain unimplemented); this experiment
   only sets the bar any persona proposal must beat.

**Judge fix, measured (same judged-pair bench)**: three iterations, each
defeating how flash-lite gamed the previous guard — (1) a count-only floor
was satisfiable while dropping a whole lane's uniques; (2) identity
accounting alone was satisfiable by lumping distinct findings into one
issue with all ids listed; (3) the shipped design adds a deterministic
merge-legality check (sources may merge only same-file within a small line
window) with verbatim split/restore repair, and a full deterministic-union
fallback for untrustworthy accounting. Scores across the same five
pairings: old contract 86% / B1 1-in-5 / 9.8 findings; count floor 97% but
3-of-5 posts were the full chattier union; **shipped identity+legality
judge: 95% recall, B1 4-in-5, 12.0 mean findings with genuine dedup**
(residual point lost to judge paraphrasing vs the keyword scorer, not to
dropped findings — judged text rewrites can under-credit keyword matching).

**Recommendation, final**: with the shipped identity+legality judge, the
v1.3 two-model roster (`models: x-ai/grok-4.6,z-ai/glm-5.3-flash`) is
deployable — **measured 95% recall / B1 4-in-5 through the full production
path** at ~1.04x cost (vs 94% / 2-in-5 single-lane). "Recall-safe" means
the deterministic coverage/legality/fallback layers structurally prevent
the judge from silently dropping or lumping away a lane's findings; the
judge's paraphrasing freedom on legal same-location merges remains, which
is where the gap to the 98% pre-judge union lives. Wall-clock note:
two-lane wall time ≈ the slower lane (GLM ~150s fixture-scale) plus the
judge call.

## 2026-08-28 — path-profile A/B: mechanism shipped, profile recommended per-repo (x-ai/grok-4.6, 5 runs/arm)

Research report #3 as a caller-owned `path_profiles` input, tested on the
HARDENED fixture (v2.1: de-coached, semantically-tight strata labels — all
earlier F1/R6 rates were retired as gameable). The tested profile targets
source-of-truth values on `*calc*` / `*rules*` paths.

| arm | recall | bug | diff-stratum | B1 | N5 | clean-twin/run |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (falsification prompt) | 94% | 70% | 92% | 2/5 | 4/5 | 4.8 |
| + source-of-truth profile | 97% | 80% | 96% | 3/5 | 5/5 | 5.6 |

The hardened baseline also settled an open question: **F1 5/5 and R6 5/5
with zero coaching** — the file/repo strata genuinely hold at 100% under
the falsification prompt; the earlier concern that those rates were
artifacts is resolved in the prompt's favor.

**Disposition**: the mechanism ships (opt-in, caller-owned, zero effect
unless configured). The measured profile is a per-repo recommendation, not
a default: the gain is real and lands exactly on the targeted
source-of-truth class (+1/5 B1, +1/5 N5, no stratum regressed), the cost
is +0.8 findings/run of extra chatter on clean PRs touching matching files.
At n=5 the B1 movement is one run — treat as directional; the live
RetireGolden registry PRs (where this class decided the bake-offs) are the
confirming measurement.

## 2026-08-28 — contract-map prompt A/B: NOT adopted (x-ai/grok-4.6, 5 runs/arm)

The research report's #2 (build an internal contract/intent map before the
sweeps), tested on fixture v2 (13 labels incl. the new file/repo-context
plants) against the falsification-prompt baseline:

| arm | recall | bug | diff-stratum | B1 | N5 | clean-twin/run | time (planted/clean) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 91% | 70% | 88% | 2/5 | 2/5 | 4.6 | 87s / 172s |
| + contract map | 91% | 70% | 88% | 2/5 | 3/5 | 5.0 | 102s / 183s |

**Not adopted**: recall identical in every stratum, the hypothesized target
(B1, the stale source-of-truth dollar) unmoved, clean-twin volume slightly
up, +10-16% wall time. The exhaustive+falsification prompt already does the
cross-file work the map was meant to add — fixture v2's validation run
showed F1 (file-context year-gate drift) and R6 (repo-context caller break)
at 5/5 under the baseline. Negative result recorded so the idea is not
re-tried without new evidence; the remaining headroom (B1/N5, both
source-of-truth-value class) points at a targeted guidance profile instead.

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

## Body-formatting instruction screening (2026-08-28, pre-1.2.5)

Change under test: the shared prompt now requires finding bodies written
for a skimming human — short paragraphs separated by blank lines
(failure scenario, then evidence, then what was checked), markdown
bullets for multi-instance findings, backticks on identifiers — with an
explicit "formatting only, never trim substance" clause. Renderer
changes (severity-emoji headings, metadata line, cost line) are
model-invisible and were not screened.

Screening: 3 runs, planted-mini (hardened, 13 labels), grok-4.6,
effort=high — the low-risk-change protocol (5+ runs reserved for
production decisions).

| metric | screen (n=3) | baseline (n=5, hardened fixture) |
| --- | --- | --- |
| recall | 90% (13/13, 11/13, 11/13) | 94% |
| bug recall | 67% (B1 1/3) | 70% (B1 2/5) |
| precision / noise | 100% / 0% | 100% / low |
| misses | B1 ×2, N5 ×2 only | B1, N5 are the two flaky labels |

Every label except B1/N5 detected 3/3; file and repo strata 100%. The
misses land exactly on the two labels with known ~40% detection
variance, so n=3 at 90% is indistinguishable from the 94% baseline — no
regression signal. Formatting effect confirmed mechanically: 28/28
findings across the three runs are multi-paragraph (previously single
dense blocks); zero bullets, as expected on single-instance fixture
plants. ADOPTED.
