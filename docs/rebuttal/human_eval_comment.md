# Human Baseline

---

**Human expert baseline on FactoryBench (Levels 1 to 3)**

We have completed a human expert baseline on a 60-item sample spanning Levels 1 to 3, and report the protocol and results below. It establishes the achievable-performance reference against which the model scores should be read, and confirms that the items are solvable from the released telemetry.

**Protocol.** A robotics expert answered 60 items under open-tool, no-LLM conditions:

- **Open tools.** The annotator was free to use any analysis instrumentation, and built a small Python toolkit for the task: series parsing, windowed condition search and threshold crossings, linear extrapolation, per-channel signal statistics, forward-kinematics consistency checks, and nominal-versus-actual absolute-difference comparison. The reference we want is what a competent engineer with normal instrumentation can extract from the telemetry, not what a person can do reading serialised numbers unaided.
- **No LLM assistance.** The annotator did not use a language model to produce or check any answer, so the baseline is independent of the systems under evaluation.
- **Identical items, identical scoring.** The annotator worked from the same items and prompts the model panel saw, and the answers were scored by the benchmark's own grader: the same acceptance bounds for scalar and vector answers, the same exact-match rule for single-select and ranking, and the same per-position partial credit on multi-select. No separate human rubric was introduced, and the human receives no credit the panel would not receive.
- **Duration.** The entire process took a total of 2 work days of question answering and tool design by the expert.

**Composition.** 13 items at Level 1, 37 at Level 2, 10 at Level 3, drawn across 14 distinct templates and covering all five answer formats. We plan on extending the baseline to address this current imbalance.

**Result.**

|                        | mean score | exact correct |
| ---------------------- | ---------- | ------------- |
| **Overall (60 items)** | **0.917**  | 49/60         |
| Level 1                | 0.962      | 12/13         |
| Level 2                | 0.980      | 34/37         |
| Level 3                | 0.625      | 3/10          |

Against the model panel on the identical items, under identical scoring:

|         | expert | Claude Sonnet 4.6 | Mistral Large 3 | GPT-5.1 | Qwen3-235B | DeepSeek V3.2 |
| ------- | ------ | ----------------- | --------------- | ------- | ---------- | ------------- |
| Level 1 | 0.962  | 0.712             | 0.500           | 0.635   | 0.519      | 0.481         |
| Level 2 | 0.980  | 0.662             | 0.444           | 0.392   | 0.385      | 0.351         |
| Level 3 | 0.625  | 0.325             | 0.550           | 0.500   | 0.575      | 0.675         |

The expert clears 0.96 at both Level 1 and Level 2 while the best model sits at 0.68 and the rest between 0.39 and 0.46. This settles the solvability question directly: the items are answerable from the released signals, so the low model scores reflect model limitations rather than broken or ambiguous items. It also shows that human experts are capable of much deeper understanding of general machines, whether in healthy or anomalous state, even though this understanding is obtained through a time-consuming process and computations (usually causing costly downtime) rather than intuitive signal comprehension.

The margin is not uniform across templates. It is narrowest where the answer can be read off a single channel with a short calculation, and widest on the templates that require committing to a specific root cause or a specific machine from the signal, where the expert reaches 1.000 and the best model only 0.429.

**Level 3 is materially harder for the expert too**, at 0.625 with 3 of 10 items exactly correct. This is generally explained by how undercovered and difficult the problem of counterfactual prediction is in time series analysis, leading to a lack of specific tools.

**Extension.** We are extending the baseline to at least 100 Q&A pairs with a minimum of 25 per level, which means adding the Level 4 decision-making questions and broadening the Level 1 and Level 3 samples. We commit to reporting the full per-level results in an updated appendix for the camera-ready version, and to releasing the baseline itself: for every item, the expert's answer, its score, and the method used to reach it.

We thank the reviewers for their valuable feedback.
