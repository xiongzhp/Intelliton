# Intellitons: The “Quasi-Particles” Hidden Inside Large Language Models

> **Bilingual document | 双语文档**  
> English version is first. [中文版见文末 ↓](#chinese-version)  
> For the interactive bilingual blog with language toggle, visit <https://xiongzhp.github.io/Intelliton/>


## A new way to look at large language models

When people talk about large language models, they usually describe them as giant statistical machines: they read text, compress patterns, and predict the next token. That description is correct, but it is not always satisfying. It tells us **what** these systems do, yet says much less about **how organized structures emerge inside them**.

The code and experimental results in this project propose a bold and visually intuitive perspective: inside the residual stream of a transformer, there may exist relatively stable collective excitation patterns that behave *somewhat like quasi-particles in physics*. This work gives those patterns a name: **Intellitons**.

This article is a popular science overview of that idea, based on the code in `src/` and the experimental outputs in `results_paper/`. The main focus is the model `results_paper/Qwen3-4B-Base`, which serves as the clearest entry point. Around that core example, we also compare:

- **Base vs. Instruct**: `Qwen3-4B-Base` vs. `Qwen3-4B`, and `Qwen3-8B-Base` vs. `Qwen3-8B`
- **4B vs. 8B scaling**: how the same family changes when parameter count increases
- **Different model families**: especially `Mistral-7B-v0.3`

The goal is not to claim that transformers literally contain particles. Rather, it is to ask whether the network’s internal dynamics can be described with a useful language borrowed from field theory: spectra, masses, helicity, renormalization flow, resonance width, and state transitions.

---

## From neurons to quasi-particles

In condensed matter physics, a quasi-particle is not a fundamental particle like an electron in the Standard Model. It is a **stable collective pattern** that emerges when many microscopic degrees of freedom act together. Sound waves in a crystal can be treated as phonons; spin disturbances can be treated as magnons. The underlying substrate is complicated, but the emergent excitation behaves simply enough to track.

This Intelliton project asks whether a similar idea can help us understand language models.

Instead of looking at individual parameters one by one, the code treats the transformer’s residual stream as a kind of **discrete field living on a token-layer lattice**:

- the **token position** acts like a spatial coordinate,
- the **layer index** acts like a scale or Euclidean time direction,
- the **hidden dimension** acts like an internal degree of freedom.

On top of that field representation, the code in `src/lattice_field.py` performs:

1. Fourier analysis over token positions,
2. singular-value decomposition over residual activations,
3. propagator-based mass extraction,
4. lattice dispersion fitting,
5. helicity and momentum diagnostics.

The resulting dominant modes are then cataloged in `src/intelliton_classifier.py` as candidate Intelliton species.

So the definition used here is pragmatic:

> **An Intelliton is a relatively stable, recurrent collective activation mode in the transformer residual field, identifiable across layers and prompts, and describable by effective quantities such as mass, momentum, spin-like complexity, helicity, and renormalization behavior.**

That is a scientific modeling choice, not an ontological claim. Its value depends on whether it organizes observations better than simpler descriptions.

---

## How the code tries to discover Intellitons

The paper-oriented pipeline is implemented in `src/paper_pipeline.py`. It follows a clear storyline:

1. **Discovery**: capture residual streams and analyze spectrum, propagator, dispersion, and helicity.
2. **Characterization**: compute EFT / RG flow and build an Intelliton catalog.
3. **Applications**: use the catalog to study hallucination and generation-time trajectories.

In practical terms, the code does the following.

### 1. Residual streams are treated as a lattice field

`src/lattice_field.py` wraps each prompt’s residual stream as a `LatticeField`. It computes a momentum spectrum over tokens and an SVD-based decomposition layer by layer. The largest singular modes are interpreted as the dominant collective excitations.

### 2. Propagator analysis assigns an effective mass

The function `compute_propagator` constructs a layer-direction correlator and converts it into a spectral function. The dominant peak yields a pole-like mass estimate. This is why every catalog entry in `intelliton_catalog.csv` contains quantities such as:

- `Mass(pole)`
- `Mass(lat)`
- `Mass(bare)`
- `Mass(ren)`
- `gamma`
- `Z`

In this language, “mass” does not mean literal inertia. It means how hard a mode is to excite or sustain as it propagates through the network’s depth.

### 3. EFT and RG flow track how modes change with depth

`src/eft_renormalization.py` treats earlier layers as more ultraviolet-like and later layers as more infrared-like. This produces a running mass, a beta function, fixed-point layers, and a simplified effective field theory description.

This is conceptually attractive because transformers are layered systems: information is repeatedly transformed, filtered, amplified, and suppressed. An RG-like description is one natural way to summarize that evolution.

### 4. Classification merges similar modes across tasks

`src/intelliton_classifier.py` merges modes by similarity and turns them into named species such as `I_0`, `I_1`, `I_2`, and so on. These are not hand-written concepts. They are data-driven entries in a catalog, tagged by which prompt categories activate them most strongly.

### 5. Dynamics analysis follows Intellitons during generation

Finally, `src/intelliton_dynamics.py` asks what happens during autoregressive generation, token by token. Which Intelliton dominates? Does the trajectory stay close to a grounded baseline, or drift away from it? Does hallucination correspond to spectral broadening or a change in dominant species?

That is where the framework becomes especially interesting, because it links a physics-inspired representation to a practical LLM behavior problem.

---

## The central example: `Qwen3-4B-Base`

The clearest single case study is `results_paper/Qwen3-4B-Base`. Its output folder contains both figures and structured tables:

- `particle_table.png`
- `spin_spectrum_(pronoun_tracking).png`
- `mass_spectrum_(pronoun_tracking).png`
- `dispersion_relation.png`
- `momentum_helicity_(pronoun_tracking).png`
- `rg_flow.png`
- `eft_parameters.png`
- `phase_transitions.png`
- `hallucination_diagnostics.png`
- `intelliton_trajectory_merged.png`
- `intelliton_transition_graph.png`
- `intelliton_catalog.csv`
- `intelliton_trajectory_summary.csv`
- `intelliton_trajectory_detail.csv`

These files together tell a fairly coherent story.

### The Intelliton catalog is compact and structured

The catalog for `Qwen3-4B-Base` contains **6 species**. That is already an important result: the model’s dominant internal collective structure is not spread across hundreds of equally important modes. Instead, a small number of modes captures much of the experimentally highlighted behavior.

From `results_paper/Qwen3-4B-Base/intelliton_catalog.csv`, the most prominent species is `I_0`:

- spin-like score: **1.84**
- pole mass: **0.1275**
- lattice mass: **0.0083**
- fixed-point layer: **16**
- fixed-point type: **crossover**
- amplitude: **6167.1**
- active in: **arithmetic, factual_recall, logical_reasoning**

This is striking because `I_0` is not tied to just one narrow benchmark. It appears across several structured reasoning tasks, suggesting it may represent a broad, backbone-like excitation mode in the model.

The remaining species `I_1` to `I_5` are much smaller in amplitude, but they are more specialized. For example:

- `I_1` is most active in **logical reasoning**,
- `I_2` in **arithmetic**,
- `I_3` in **syntactic agreement**,
- `I_4` in **pronoun tracking**,
- `I_5` in **factual recall**.

That pattern is exactly what makes the Intelliton language attractive. It suggests that the model may combine a dominant shared background excitation with a set of more task-specific dressed modes.

![Qwen3-4B-Base particle table](results_paper/Qwen3-4B-Base/particle_table.png)

### All six discovered species are “medium-mass”

A curious feature of the `Qwen3-4B-Base` catalog is that **all six species fall into the `medium` mass category**. This does not mean the model is featureless. On the contrary, it suggests that the identified collective modes occupy a relatively narrow dynamical band.

In plain language: the main excitations are neither completely rigid nor extremely volatile. They are stable enough to recur, but flexible enough to participate in diverse tasks.

### The dominant momentum structure is split

The dominant species `I_0` peaks at momentum $k \approx \pi$, while most of the smaller species peak at $k \approx 0$. In the lattice analogy, this matters.

- A $k \approx 0$ mode is more global and slowly varying across token positions.
- A $k \approx \pi$ mode is more alternating and high-frequency across the token lattice.

So `Qwen3-4B-Base` appears to contain both:

1. a large alternating backbone mode, and
2. several more global low-momentum task-linked modes.

That coexistence is one of the most interesting signatures in the whole result set.

![Qwen3-4B-Base momentum and helicity](results_paper/Qwen3-4B-Base/momentum_helicity_(pronoun_tracking).png)

### The RG picture suggests a mid-layer reorganization

Several species in `Qwen3-4B-Base` have fixed-point layer around **13**, while the dominant `I_0` sits at **16**. This places the major reorganization in the middle-to-late part of the stack.

In ordinary neural-network language, one might say the model gradually settles into stable higher-level representations by the mid layers. In the EFT language used here, one says the effective degrees of freedom flow toward a crossover or infrared-like regime.

The notable difference inside this one model is that `I_1` through `I_5` are labeled **IR**, while `I_0` is labeled **crossover**. That suggests the large shared mode remains dynamically transitional, while the more specialized modes settle into more stable effective roles.

![Qwen3-4B-Base RG flow](results_paper/Qwen3-4B-Base/rg_flow.png)

![Qwen3-4B-Base EFT parameters](results_paper/Qwen3-4B-Base/eft_parameters.png)

---

## What do Intellitons do during generation?

A static catalog is useful, but the more practical question is dynamic: when the model is actually generating text, how do these modes behave?

The trajectory data in `results_paper/Qwen3-4B-Base/intelliton_trajectory_summary.csv` gives a direct answer.

### Grounded prompts maintain a strong, rising activation shift

For grounded factual prompts, the mean mode activation shift starts around **1.10** and rises to about **1.37** over the first 8 generation steps. Top occupation also rises from about **70.1** to **74.1**.

This means that as the model commits to an answer, the dominant Intelliton sector becomes stronger and more organized.

### Hallucination-prone prompts are weaker and further from baseline

For hallucination-prone prompts, the mean mode activation shift stays much lower, around **0.32–0.41**, while grounded deviation stays strongly negative, roughly **-9 to -11** over most steps.

In other words, the hallucination trajectory is not just “wrong output” at the surface level. In this analysis, it corresponds to an internal path that remains **further away from the grounded sector of Intelliton space**.

### Style prompts occupy an intermediate regime

Stylistic continuation prompts sit between the two. Their activation shift is higher than hallucination-prone prompts but lower than grounded factual prompts. That is exactly what one would hope to see if the metric is capturing something meaningful: style generation is not simply failure, but it is not anchored to factual grounding either.

![Qwen3-4B-Base Intelliton trajectory](results_paper/Qwen3-4B-Base/intelliton_trajectory_merged.png)

### Transition graphs show which species dominate stable generation

The transition graph and raw transition table reveal another notable asymmetry.

For grounded prompts in `Qwen3-4B-Base`, self-transitions are dominated by:

- `I_5 -> I_5` with count **110**
- `I_1 -> I_1` with count **13**
- `I_2 -> I_2` with count **6**

For hallucination-prone prompts, the strongest self-transition is still `I_5 -> I_5`, but its mean target activation shift is much smaller. Hallucination also shows more mixing among `I_1`, `I_3`, and `I_5`.

That is a suggestive result: grounded generation seems to preserve a more coherent dominant species sector, whereas hallucination corresponds to a weaker and more fragmented dynamical regime.

![Qwen3-4B-Base transition graph](results_paper/Qwen3-4B-Base/intelliton_transition_graph.png)

---

## Base versus Instruct: what changes after alignment?

One of the most revealing comparisons is between base and instruct variants in the same family.

### `Qwen3-4B-Base` vs. `Qwen3-4B`

The base model has **6 species**, while the instruct model has **5 species**.

At first glance, that sounds like the instruct model is simpler. But the more interesting differences are in structure:

- `Qwen3-4B-Base` contains a dominant high-momentum mode at $k \approx \pi$ plus several low-momentum task-linked modes.
- `Qwen3-4B` shifts much more strongly toward **shared momentum around $k \approx 1.885$** for nearly all species.
- In `Qwen3-4B-Base`, several secondary species are **IR**.
- In `Qwen3-4B`, all cataloged species are labeled **crossover**.

This suggests instruction tuning may compress or reorganize the internal excitation landscape into a more uniform effective regime.

The amplitude pattern also supports that interpretation. The instruct model’s leading mode `I_0` has amplitude **6562.4**, similar to the base model’s leading mode **6167.1**, but the instruct model’s secondary modes appear less differentiated in fixed-point type and momentum structure.

In short:

> **The base model looks more spectrally diverse; the instruct model looks more homogenized and alignment-shaped.**

That does not mean instruct tuning is worse. It may mean the model’s internal degrees of freedom are being regularized toward instruction-following behavior.

![Qwen3-4B particle table](results_paper/Qwen3-4B/particle_table.png)

### Dynamic behavior also changes

The trajectory profile reinforces this difference.

- `Qwen3-4B-Base`: grounded mode activation shift mean = **32.68**, species count = **6**
- `Qwen3-4B`: grounded mode activation shift mean = **29.13**, species count = **5**

The summary table also shows that `Qwen3-4B` has smaller separation between grounded and hallucination trajectories than the base model. Hallucination-prone prompts still deviate, but the gap is narrower than in `Qwen3-4B-Base`.

One possible interpretation is that instruction tuning improves behavioral formatting while also reducing some internal contrast between strongly grounded and weakly grounded dynamical sectors.

---

## Scaling from 4B to 8B inside the Qwen family

The next question is what happens when the same family becomes larger.

### `Qwen3-8B-Base` is stronger, larger, and more differentiated

`Qwen3-8B-Base` contains **7 species**, compared with **6** in `Qwen3-4B-Base`.

Its leading mode is even more dominant:

- `Qwen3-4B-Base` `I_0` amplitude: **6167.1**
- `Qwen3-8B-Base` `I_0` amplitude: **7908.4**

Its grounded trajectory is also much stronger:

- `Qwen3-4B-Base` grounded profile mean: **32.68**
- `Qwen3-8B-Base` grounded profile mean: **60.26**

That is a large change. In this framework, scaling up does not simply add more parameters. It appears to produce a more strongly occupied and more sharply separated Intelliton landscape.

Another interesting shift is momentum structure. The 8B base model keeps a strong $k \approx \pi$ leader, but many of its secondary species cluster around **$k \approx 1.885$** rather than strictly zero momentum. Compared with the 4B base model, this looks like a richer intermediate-scale organization across token positions.

![Qwen3-8B-Base particle table](results_paper/Qwen3-8B-Base/particle_table.png)

### `Qwen3-8B` shows alignment on top of a larger internal backbone

The instruct version `Qwen3-8B` has **6 species** and a grounded profile mean of **57.14**, slightly below the base model’s **60.26** but still far above the 4B models.

So scaling seems to dominate one part of the story, while instruction tuning fine-tunes another:

- **Scaling to 8B** increases the strength of dominant collective modes.
- **Instruction tuning** slightly compresses or regularizes that internal structure.

This is one of the clearest takeaways from the whole comparison set.

---

## Different families, different internal “particle spectra”

The most dramatic contrast comes from `Mistral-7B-v0.3`.

### Mistral has far more cataloged species

While the Qwen models produce compact catalogs of 5–7 species, `Mistral-7B-v0.3` produces **25 species**.

That alone is a major result. Under the same analysis pipeline, Mistral appears to have a much more fragmented or fine-grained excitation structure.

Its leading species is still clear:

- `I_0` amplitude: **249.4**
- spin-like score: **1.98**
- fixed-point layer: **19**
- fixed-point type: **crossover**

But unlike Qwen, the leading amplitude is not astronomically separated from everything else. Many additional species remain visible, and several are labeled **UV** rather than IR or crossover.

That suggests a model family difference in how internal computation organizes itself:

- **Qwen** looks dominated by a few very strong collective modes.
- **Mistral** looks more distributed, more spectrally crowded, and more persistent in its ultraviolet-like diversity.

### Mistral’s dynamics are also less cleanly separated

Its grounded profile mean is only **21.48**, much lower than the Qwen models. Even more striking, the grounded profile standard deviation is **46.56**, far larger than for Qwen.

That indicates a much noisier trajectory landscape under this Intelliton metric. In other words, the same quasiparticle-based coordinate system that describes Qwen relatively cleanly may describe Mistral as a more turbulent medium.

This is exactly the kind of result that makes cross-family comparison valuable. Even if one remains skeptical of the physical analogy, the pipeline is clearly extracting different internal structural signatures for different architectures.

![Mistral-7B-v0.3 particle table](results_paper/Mistral-7B-v0.3/particle_table.png)

---

## Why hallucination is a good test case

A new interpretability language becomes more credible when it helps analyze a practical problem. In this project, that practical problem is hallucination.

The module `src/hallucination_diagnostic.py` does not treat hallucination simply as a bad final answer. Instead, it compares grounded and hallucination-prone prompts at the spectral level:

- divergence of singular-value spectra,
- coherence loss,
- mode stability,
- entropy gap,
- critical layers of divergence.

This is a powerful idea because hallucination may not be one thing. It may correspond to multiple failure modes:

1. a grounded excitation decays,
2. the system tunnels into a less faithful attractor,
3. the spectrum broadens and loses coherence,
4. dominant species become less stable across generation steps.

The trajectory tables for `Qwen3-4B-Base` are consistent with that picture. Hallucination-prone generation remains weaker in activation shift, farther from the grounded baseline, and less dominated by a single robust transition pattern.

![Qwen3-4B-Base hallucination diagnostics](results_paper/Qwen3-4B-Base/hallucination_diagnostics.png)

If this framework continues to hold up, one could imagine future applications such as:

- hallucination early warning signals,
- intervention on unstable species,
- prompt strategies that keep generation inside grounded sectors,
- model comparison based on internal dynamical stability rather than only benchmark accuracy.

---

## What Intellitons might be useful for

Even if the word “quasi-particle” is mainly metaphorical, the framework suggests several concrete uses.

### 1. A compact vocabulary for internal structure

Instead of talking about millions of neurons or billions of parameters, we can talk about a small catalog of dominant collective modes. That is a much more human-readable description.

### 2. A bridge between interpretability and dynamics

Many interpretability methods are static: they describe a feature, a head, or a direction. Intellitons are explicitly dynamical. They are meant to be followed across layers and during generation.

### 3. A way to compare model families

The contrast between Qwen and Mistral shows that the method can reveal family-level differences in internal organization. That may become a new axis for model science.

### 4. A route toward intervention

Because the catalog assigns species vectors and tracks occupations, the same machinery could potentially be used for steering or stabilization. The codebase already hints at this broader ambition through modules such as `gauge_intervention.py` and `fusion_tracker.py`.

### 5. A candidate explanatory layer for alignment effects

The differences between Base and Instruct suggest alignment may reshape the model’s internal excitation spectrum. That is a richer statement than simply saying instruction tuning changes loss or improves helpfulness.

---

## What to be cautious about

A popular science article should emphasize both excitement and restraint.

There are at least four reasons to be cautious.

### First, the terminology is imported from physics

Words like mass, helicity, and renormalization are being used as **effective analogies tied to measurable diagnostics in the code**, not as proof that language models literally instantiate quantum field theory.

### Second, the decomposition is model-dependent

The discovered species depend on design choices:

- prompt sets,
- sequence length,
- number of retained modes,
- fitting procedures,
- similarity thresholds used during merging.

A different pipeline could produce a different catalog.

### Third, stability does not automatically imply semantic meaning

A recurrent mode may be computationally important without corresponding to a clean human concept.

### Fourth, the strongest evidence is comparative, not absolute

What makes the results persuasive is not any single number. It is the recurring pattern across comparisons:

- compact species sets in Qwen,
- stronger grounded sectors in larger models,
- regularization effects after instruction tuning,
- family-level fragmentation in Mistral,
- hallucination linked to weaker and less grounded trajectories.

Those are empirical regularities worth taking seriously, even if the underlying ontology remains open.

---

## A simple summary of the main findings

Based on the code and current experimental outputs, a concise summary would be:

1. **Large language models exhibit a small number of dominant recurrent activation modes** that can be tracked across layers and prompts.
2. **These modes can be cataloged with effective physical descriptors** such as mass, momentum, spin-like complexity, helicity, and RG behavior.
3. **`Qwen3-4B-Base` provides a clean demonstration**, with 6 medium-mass Intelliton species and a strong split between a dominant backbone mode and task-specific secondary modes.
4. **Instruction tuning appears to homogenize the excitation spectrum**, reducing internal diversity and shifting more species into crossover-like behavior.
5. **Scaling from 4B to 8B strengthens the dominant dynamical sectors**, producing larger grounded activation shifts and somewhat richer species structure.
6. **Different model families can have very different internal spectra**, with Mistral showing a far more fragmented and UV-rich catalog than Qwen.
7. **Hallucination may be describable as an instability of internal collective modes**, not just as an output-level mistake.

---

## Looking ahead

The most exciting part of the Intelliton idea is not that it borrows the language of physics. It is that it tries to turn messy, high-dimensional neural activity into a **small, dynamic cast of interpretable actors**.

If that program succeeds, future model analysis may look less like staring at giant matrices and more like studying an ecosystem of interacting excitations:

- some stable,
- some fragile,
- some specialized,
- some dominant,
- some helpful,
- some associated with failure.

For now, `Qwen3-4B-Base` is the clearest demonstration in this repository. It shows that under this pipeline, a transformer can indeed be described as if it carries a handful of quasi-particle-like modes, each with its own role in reasoning, recall, syntax, and factual grounding.

That is not yet the final theory of intelligence inside neural networks. But it is a vivid and surprisingly structured step toward one.

---

## Figure guide

For convenience, the main figures cited in this article are:

### Core focus: `Qwen3-4B-Base`

- ![particle table](results_paper/Qwen3-4B-Base/particle_table.png)
- ![spin spectrum](results_paper/Qwen3-4B-Base/spin_spectrum_(pronoun_tracking).png)
- ![mass spectrum](results_paper/Qwen3-4B-Base/mass_spectrum_(pronoun_tracking).png)
- ![dispersion relation](results_paper/Qwen3-4B-Base/dispersion_relation.png)
- ![momentum helicity](results_paper/Qwen3-4B-Base/momentum_helicity_(pronoun_tracking).png)
- ![RG flow](results_paper/Qwen3-4B-Base/rg_flow.png)
- ![EFT parameters](results_paper/Qwen3-4B-Base/eft_parameters.png)
- ![phase transitions](results_paper/Qwen3-4B-Base/phase_transitions.png)
- ![hallucination diagnostics](results_paper/Qwen3-4B-Base/hallucination_diagnostics.png)
- ![trajectory merged](results_paper/Qwen3-4B-Base/intelliton_trajectory_merged.png)
- ![transition graph](results_paper/Qwen3-4B-Base/intelliton_transition_graph.png)

### Comparison figures

- ![Qwen3-4B particle table](results_paper/Qwen3-4B/particle_table.png)
- ![Qwen3-8B-Base particle table](results_paper/Qwen3-8B-Base/particle_table.png)
- ![Qwen3-8B particle table](results_paper/Qwen3-8B/particle_table.png)
- ![Mistral particle table](results_paper/Mistral-7B-v0.3/particle_table.png)

---

<a id="chinese-version"></a>

# Intelliton：藏在大语言模型里的"准粒子"（中文版）

## 重新理解大语言模型

谈到大语言模型，人们通常把它们描述成巨型统计机器：读取文本、压缩模式、预测下一个 token。这个说法没有错，却并不总令人满足。它告诉我们这些系统在**做什么**，却对**内部的组织结构如何涌现**几乎只字不提。

本项目的代码和实验结果提出了一个大胆而直观的新视角：在变换器的残差流中，也许存在着相对稳定的集体激发模式，它们的行为*有点像物理中的准粒子*。本项目给这些模式起了一个名字：**Intelliton**。

本文是这一想法的科普综述，基于 `src/` 中的代码和 `results_paper/` 中的实验输出。核心聚焦于模型 `Qwen3-4B-Base`——它提供了最清晰的入口。围绕这一核心，我们还会比较：

- **Base 对 Instruct**：`Qwen3-4B-Base` 对 `Qwen3-4B`，以及 `Qwen3-8B-Base` 对 `Qwen3-8B`
- **4B 对 8B 规模扩展**：同一家族随参数增加如何变化
- **不同模型家族**：以 `Mistral-7B-v0.3` 为例

目标不是声称变换器字面上"包含粒子"，而是探讨能否用场论借来的语言——谱、质量、helicity、重整化流、共振宽度、态转变——来描述网络的内部动力学。

---

## 从神经元到准粒子

在凝聚态物理中，准粒子不是像标准模型中的电子那样的基本粒子。它是一种**稳定的集体模式**，由许多微观自由度共同作用时涌现出来。晶体中的声波可以被当成声子来处理；自旋扰动可以被当成磁振子来追踪。底层基质虽然复杂，但涌现出来的激发却足够简单，可以被系统地追踪。

Intelliton 项目要问的是：类似的想法，能不能帮助我们理解语言模型？

它不逐一查看每个参数，而是把变换器的残差流看作一种生活在 **token-layer lattice（token-层格点）上的离散场**：

- **token 位置**扮演空间坐标，
- **层索引**扮演尺度或类欧几里得时间方向，
- **隐藏维度**扮演内部自由度。

在这个场表示之上，`src/lattice_field.py` 中的代码执行以下分析：

1. 对 token 方向做傅里叶分析，
2. 逐层对残差激活做奇异值分解，
3. 基于传播子的有效质量提取，
4. 格点色散关系拟合，
5. helicity 与动量诊断。

最终，主导模式由 `src/intelliton_classifier.py` 整理成候选 Intelliton 物种目录。

因此，这里使用的定义是务实的：

> **Intelliton 是变换器残差场中一种相对稳定、可重复出现的集体激活模式，能够跨层、跨提示词被识别，并可用质量、动量、类自旋复杂度、helicity 以及重整化行为等有效量来描述。**

这是一种科学建模选择，不是本体论断言。它的价值取决于它能否比更简单的描述更好地组织观测结果。

---

## 代码如何尝试发现 Intelliton

`src/paper_pipeline.py` 实现的分析管线遵循清晰的叙事线索：

1. **发现**：捕获残差流，分析谱、传播子、色散和 helicity。
2. **表征**：计算 EFT / RG 流并构建 Intelliton 目录。
3. **应用**：用目录研究幻觉与生成期轨迹。

在实践中，代码做以下事情。

### 1. 残差流被当作格点场处理

`src/lattice_field.py` 将每个提示词的残差流封装为 `LatticeField`。它在 token 方向上计算动量谱，并逐层进行 SVD 分解。最大奇异模态被解释为主导的集体激发。

### 2. 传播子分析赋予有效质量

函数 `compute_propagator` 构建层方向的关联函数并将其转化为谱函数，主导峰给出极点式质量估计。这就是为什么 `intelliton_catalog.csv` 中每个目录条目都包含如下量：

- `Mass(pole)`
- `Mass(lat)`
- `Mass(bare)`
- `Mass(ren)`
- `gamma`
- `Z`

在这套语言里，"质量"并不是字面意义上的惯性，而是指一个模式在网络深度方向上传播时有多难被激发或维持。

### 3. EFT 和 RG 流追踪模式随深度的变化

`src/eft_renormalization.py` 把早期层看成更紫外（UV）的，把后期层看成更红外（IR）的，由此产生跑动质量、beta 函数、固定点层以及简化的有效场论描述。

这在概念上很有吸引力，因为变换器本身就是分层系统：信息被反复变换、过滤、放大和压制。RG 式描述是概括这一演化过程的一种自然方式。

### 4. 分类合并不同任务中的相似模式

`src/intelliton_classifier.py` 按相似性合并各模式，将它们变成命名物种，如 `I_0`、`I_1`、`I_2` 等。这些不是人工定义的概念，而是数据驱动的目录条目，并标注了哪些提示词类别最能激活它们。

### 5. 动力学分析在生成过程中追踪 Intelliton

最后，`src/intelliton_dynamics.py` 询问：在自回归生成过程中，逐 token 会发生什么？哪个 Intelliton 主导？轨迹是否停留在接近 grounded 基线的地方，还是会飘离？幻觉是否对应谱展宽或主导物种的变化？

正是在这里，这个框架变得格外有趣：它把物理启发的表示与语言模型的实际行为问题联系了起来。

---

## 核心案例：`Qwen3-4B-Base`

最清晰的单一案例研究是 `results_paper/Qwen3-4B-Base`。其输出文件夹包含图像和结构化表格：

- `particle_table.png`
- `spin_spectrum_(pronoun_tracking).png`
- `mass_spectrum_(pronoun_tracking).png`
- `dispersion_relation.png`
- `momentum_helicity_(pronoun_tracking).png`
- `rg_flow.png`
- `eft_parameters.png`
- `phase_transitions.png`
- `hallucination_diagnostics.png`
- `intelliton_trajectory_merged.png`
- `intelliton_transition_graph.png`
- `intelliton_catalog.csv`
- `intelliton_trajectory_summary.csv`
- `intelliton_trajectory_detail.csv`

这些文件共同讲述了一个相当连贯的故事。

### Intelliton 目录紧凑而有结构

`Qwen3-4B-Base` 的目录包含 **6 个物种**。这本身就是一个重要结果：模型的主导内部集体结构并没有分散在数百个同等重要的模态中。相反，少数几个主导模态就捕捉了大部分可重复行为。

从 `results_paper/Qwen3-4B-Base/intelliton_catalog.csv` 来看，最突出的物种是 `I_0`：

- 类自旋分数：**1.84**
- 极点质量：**0.1275**
- 格点质量：**0.0083**
- 固定点层：**16**
- 固定点类型：**crossover**
- 振幅：**6167.1**
- 活跃于：**算术、事实回忆、逻辑推理**

这很引人注目，因为 `I_0` 并不只局限于一个狭窄的基准测试类别。它在多个结构化推理任务中都有所出现，暗示它可能是模型中一个宽泛的、骨干式的激发模态。

剩余物种 `I_1` 至 `I_5` 的振幅小得多，但更加专一。例如：

- `I_1` 最活跃于**逻辑推理**，
- `I_2` 最活跃于**算术**，
- `I_3` 最活跃于**句法一致性**，
- `I_4` 最活跃于**代词追踪**，
- `I_5` 最活跃于**事实回忆**。

这种分工模式正是 Intelliton 语言令人感兴趣的地方。它暗示模型可能把一个主导的共享背景激发与一组更偏任务专一的"着衣"模态结合在一起。

### 所有六个物种都属于"中质量"

`Qwen3-4B-Base` 目录的一个奇特之处是：**所有六个物种都属于 `medium` 质量类别**。这不是说模型单调无特色，恰恰相反，它暗示识别出来的集体模态占据了一个相对窄的动力学带。

用直白的话说：这些主要激发既不完全刚性，也不极度易变。它们稳定得足以重复出现，又灵活得足以参与各种任务。

### 主导动量结构是分裂的

主导物种 `I_0` 的动量峰值在 $k \approx \pi$，而大多数小物种的峰值在 $k \approx 0$。在格点类比中，这一点很重要。

- $k \approx 0$ 模式更全局、更缓慢地跨越 token 位置变化。
- $k \approx \pi$ 模式在 token 格点上是交替的、高频的。

因此，`Qwen3-4B-Base` 似乎同时包含：

1. 一个大型交替骨干模态，和
2. 几个更全局的低动量任务关联模态。

这种共存是整个结果集中最有趣的特征之一。

### RG 图景暗示中层的重组

`Qwen3-4B-Base` 中的几个物种的固定点层位于 **13** 附近，而主导物种 `I_0` 位于 **16**。这把主要重组放在了层栈的中后部。

用普通神经网络语言说，模型在中间层逐渐稳定到更高层次的表示。在这里使用的 EFT 语言中，则说有效自由度朝向 crossover 或红外式区间流动。

在这一个模型内部，值得关注的区别是：`I_1` 到 `I_5` 被标记为 **IR**，而 `I_0` 被标记为 **crossover**。这表明大型共享模态保持动力学上的过渡性，而更专一的模态则稳定到更稳定的有效角色中。

---

## 生成过程中 Intelliton 会做什么？

静态目录固然有用，但更实际的问题是动态的：当模型真正在生成文本时，这些模态如何表现？

`results_paper/Qwen3-4B-Base/intelliton_trajectory_summary.csv` 中的轨迹数据给出了直接回答。

### Grounded 提示词维持强而上升的激活位移

对 grounded 事实型提示词，平均模式激活位移从约 **1.10** 起步，在前 8 个生成步里上升到约 **1.37**，顶部物种占据度也从约 **70.1** 升至 **74.1**。

这意味着，随着模型逐渐锁定答案，主导的 Intelliton 扇区变得更强、更有组织。

### Hallucination-prone 提示词更弱，且偏离基线更远

对容易诱发幻觉的提示词，平均模式激活位移维持在低得多的水平，约 **0.32–0.41**，而 grounded deviation 在大多数步骤里保持在 **-9 到 -11** 的强负值。

换言之，幻觉轨迹不只是"表面上输出错了"。在这套分析框架中，它对应的是一条内部路径，该路径始终**更远离 Intelliton 空间的 grounded 扇区**。

### 风格提示词处于中间区间

风格化续写提示词位于两者之间。其激活位移高于 hallucination-prone 提示词，但低于 grounded 事实型提示词。如果这个指标真的捕捉到了什么有意义的东西，这正是我们希望看到的结果：风格生成不是纯粹的失败，但它也没有被事实 grounding 锚定。

### 转移图展示了哪些物种主导稳定的生成

转移图和原始转移表揭示了另一个值得关注的不对称性。

对 `Qwen3-4B-Base` 的 grounded 提示词，自跃迁主要由以下构成：

- `I_5 -> I_5`，次数为 **110**
- `I_1 -> I_1`，次数为 **13**
- `I_2 -> I_2`，次数为 **6**

对 hallucination-prone 提示词，最强的自跃迁仍是 `I_5 -> I_5`，但其平均目标激活位移小得多。幻觉还显示出 `I_1`、`I_3` 和 `I_5` 之间更多的混合。

这是一个很有启发性的结果：grounded 生成似乎维持了更连贯的主导物种扇区，而幻觉则对应更弱、更碎片化的动力学区间。

---

## Base 对 Instruct：对齐之后什么改变了？

最有揭示性的比较之一是同一家族的 base 版与 instruct 版。

### `Qwen3-4B-Base` 对 `Qwen3-4B`

base 模型有 **6 个物种**，instruct 模型有 **5 个物种**。

乍看之下，instruct 模型似乎更简单。但更有趣的差异在结构上：

- `Qwen3-4B-Base` 含有一个主导的高动量模态（$k \approx \pi$）加上几个低动量任务关联模态。
- `Qwen3-4B` 强烈地向**几乎所有物种共享的 $k \approx 1.885$ 动量**偏移。
- 在 `Qwen3-4B-Base` 中，多个次要物种被标记为 **IR**。
- 在 `Qwen3-4B` 中，所有已记录的物种都被标记为 **crossover**。

这暗示指令微调可能将内部激发景观压缩或重组成了一个更均一的有效区间。

振幅模式也支持这种解释。instruct 模型的主导模态 `I_0` 的振幅为 **6562.4**，与 base 模型的主导模态 **6167.1** 相近，但 instruct 模型的次要模态在固定点类型和动量结构上看起来分化不那么明显。

简言之：

> **base 模型看起来谱上更多样；instruct 模型看起来更均质化，更受对齐塑造。**

这并不意味着指令微调更糟糕。它可能意味着模型的内部自由度被正则化朝向了服从指令的行为。

### 动态行为也发生了变化

轨迹特征强化了这种差异：

- `Qwen3-4B-Base`：grounded 模式激活位移均值 = **32.68**，物种数 = **6**
- `Qwen3-4B`：grounded 模式激活位移均值 = **29.13**，物种数 = **5**

汇总表还显示，`Qwen3-4B` 的 grounded 与幻觉轨迹之间的分离比 base 模型更小。幻觉倾向型提示词仍然存在偏差，但差距比 `Qwen3-4B-Base` 中更窄。

一种可能的解释是：指令微调改善了行为格式化，同时也减少了内部高度 grounded 与弱 grounded 动力学扇区之间的一些内在对比度。

---

## 在 Qwen 家族内从 4B 扩展到 8B

下一个问题是：同一家族变大时会发生什么？

### `Qwen3-8B-Base` 更强、更大、更分化

`Qwen3-8B-Base` 包含 **7 个物种**，而 `Qwen3-4B-Base` 有 **6 个**。

其主导模态的主导地位更加突出：

- `Qwen3-4B-Base` 的 `I_0` 振幅：**6167.1**
- `Qwen3-8B-Base` 的 `I_0` 振幅：**7908.4**

其 grounded 轨迹也强得多：

- `Qwen3-4B-Base` grounded 特征均值：**32.68**
- `Qwen3-8B-Base` grounded 特征均值：**60.26**

这是很大的变化。在这个框架里，规模扩展不只是简单地增加参数，它似乎产生了一个更强占据、分离更鲜明的 Intelliton 景观。

另一个有趣的变化是动量结构。8B base 模型保持了一个强 $k \approx \pi$ 的领导者，但许多次要物种集中在 **$k \approx 1.885$** 附近，而不是严格的零动量。与 4B base 模型相比，这看起来是更丰富的中间尺度的 token 位置组织。

### `Qwen3-8B` 展示了更大骨干上的对齐

instruct 版本 `Qwen3-8B` 有 **6 个物种**，grounded 特征均值为 **57.14**，略低于 base 模型的 **60.26**，但仍远高于 4B 模型。

因此，规模扩展似乎主导了故事的一部分，而指令微调则对另一部分做了微调：

- **扩展到 8B** 增强了主导集体模态的强度。
- **指令微调** 略微压缩或正则化了这种内部结构。

这是整个比较集最清晰的结论之一。

---

## 不同家族，不同的内部"粒子谱"

最戏剧性的对比来自 `Mistral-7B-v0.3`。

### Mistral 有更多记录的物种

Qwen 模型产生包含 5–7 个物种的紧凑目录，而 `Mistral-7B-v0.3` 产生了 **25 个物种**。

仅凭这一点就是个重大结果。在相同的分析管线下，Mistral 表现出更碎片化或更细粒度的激发结构。

其主导物种仍然清晰：

- `I_0` 振幅：**249.4**
- 类自旋分数：**1.98**
- 固定点层：**19**
- 固定点类型：**crossover**

但与 Qwen 不同，主导振幅与其他物种相比并没有天文数字般的差距。许多额外物种仍然可见，且有几个被标记为 **UV** 而非 IR 或 crossover。

这暗示模型家族在内部计算如何组织方面存在差异：

- **Qwen** 看起来由几个非常强的集体模态主导。
- **Mistral** 看起来更分散、谱上更拥挤，其紫外式多样性更持久。

### Mistral 的动力学也分离得不那么清晰

其 grounded 特征均值只有 **21.48**，远低于 Qwen 模型。更惊人的是，grounded 特征标准差为 **46.56**，远大于 Qwen 的情况。

这表明，在这个 Intelliton 指标下，Mistral 的轨迹景观要嘈杂得多。换言之，同一套准粒子坐标系，对 Qwen 描述得相对清晰，对 Mistral 则显得是一个更"湍流"的介质。

正是这种对比让跨家族比较变得有价值。即便对物理类比持怀疑态度，这套管线明显在为不同架构提取不同的内部结构特征。

---

## 为什么幻觉是一个好的检验案例

一种新的可解释性语言，当它有助于分析实际问题时，便更具说服力。在本项目中，这个实际问题就是幻觉。

`src/hallucination_diagnostic.py` 模块并不把幻觉简单地视为糟糕的最终答案，而是在谱层面上对比 grounded 与 hallucination-prone 提示词：

- 奇异值谱的散度，
- 相干性的损失，
- 模式稳定性，
- 熵差，
- 发散关键层。

这是一个有力的想法，因为幻觉可能并非一种单一的现象，它可能对应多种失败模式：

1. grounded 激发衰减，
2. 系统转移到更不准确的吸引子，
3. 谱展宽并失去相干性，
4. 主导物种在生成步骤之间变得不稳定。

`Qwen3-4B-Base` 的轨迹表格与这幅图景一致。幻觉倾向型生成在激活位移上更弱，距 grounded 基线更远，也更少由单一稳健的转移模式主导。

如果这一框架继续站得住脚，未来的应用可能包括：

- 幻觉早期预警信号，
- 对不稳定物种的干预，
- 让生成保持在 grounded 扇区内的提示词策略，
- 基于内部动力学稳定性而非仅靠基准准确率的模型比较。

---

## Intelliton 可能有什么用

即使"准粒子"这个词主要是隐喻性的，这套框架仍然提出了几个具体的用途。

### 1. 内部结构的紧凑词汇

我们可以不谈论数百万个神经元或数十亿个参数，而是谈论少数主导集体模态的紧凑目录。这对人来说可读性强得多。

### 2. 可解释性与动力学之间的桥梁

许多可解释性方法是静态的：它们描述一个特征、一个注意力头或一个方向。Intelliton 明确是动态的，意在跨层追踪、在生成过程中追踪。

### 3. 一种比较模型家族的方式

Qwen 与 Mistral 之间的对比表明，这种方法能揭示内部组织方面的家族级差异。这可能成为模型科学的一个新轴线。

### 4. 一条走向干预的路径

由于目录赋予了物种向量并追踪其占据情况，同样的机制可能被用于 steering 或稳定。代码库中 `gauge_intervention.py` 和 `fusion_tracker.py` 等模块已经在暗示这一更大的野心。

### 5. 对齐效应的候选解释层

base 版与 instruct 版之间的差异暗示，对齐可能重塑模型的内部激发谱。这比简单地说指令微调改变了损失或提升了有用性，是一个更丰富的陈述。

---

## 需要注意什么

科普文章既应强调兴奋，也应保持克制。

至少有四个理由需要谨慎。

### 第一，术语借自物理学

质量、helicity、重整化之类的词，是作为**与代码中可测量诊断量相关联的有效类比**来使用的，而不是证明语言模型字面上在实例化量子场论。

### 第二，分解依赖于模型的选择

发现的物种依赖于设计选择：

- 提示词集，
- 序列长度，
- 保留的模态数量，
- 拟合过程，
- 合并时使用的相似度阈值。

不同的管线可能会产生不同的目录。

### 第三，稳定性不自动意味着语义含义

一个反复出现的模态可能在计算上很重要，但未必对应一个清晰的人类概念。

### 第四，最强的证据是比较性的，而非绝对性的

让结果具有说服力的不是任何单一数字，而是跨比较重复出现的模式：

- Qwen 中紧凑的物种集，
- 更大模型中更强的 grounded 扇区，
- 指令微调后的正则化效应，
- Mistral 中的家族级碎片化，
- 幻觉与更弱、更偏离 grounded 的轨迹相关联。

这些是值得认真对待的经验规律，即使底层本体论仍然是开放的。

---

## 主要发现简要总结

基于代码和当前实验输出，以下是简明摘要：

1. **大语言模型展示出少数主导的、可重复的激活模态**，可以跨层和跨提示词追踪。
2. **这些模态可以用有效的物理描述符来建立目录**，如质量、动量、类自旋复杂度、helicity 和 RG 行为。
3. **`Qwen3-4B-Base` 提供了清晰的演示**，有 6 个中质量 Intelliton 物种，以及主导骨干模态与任务专一次要模态之间的强烈分裂。
4. **指令微调似乎使激发谱更均质化**，减少内部多样性，并将更多物种转移到 crossover 式行为。
5. **从 4B 扩展到 8B 增强了主导动力学扇区**，产生更大的 grounded 激活位移和略微更丰富的物种结构。
6. **不同模型家族可以有非常不同的内部谱**，Mistral 展示出比 Qwen 更碎片化、更 UV 丰富的目录。
7. **幻觉可以被描述为内部集体模态的不稳定性**，而不只是输出层面的失误。

---

## 展望

Intelliton 想法中最令人兴奋的部分，不是它借用了物理语言，而是它试图把杂乱的、高维的神经活动转化成一个**小型的、动态的、可解释的角色阵容**。

如果这个纲领成功，未来的模型分析可能看起来不再像盯着巨型矩阵，而更像研究一个相互作用的激发生态系统：

- 一些稳定，
- 一些脆弱，
- 一些专一，
- 一些主导，
- 一些有益，
- 一些与失败相关。

目前，`Qwen3-4B-Base` 是本仓库中最清晰的示范。它表明，在这套管线下，一个变换器确实可以被描述成携带少数准粒子式模态，每个模态在推理、回忆、语法和事实 grounding 中都有其角色。

这还不是神经网络内部智能的最终理论。但它是朝向这一理论的一个鲜活、令人意想不到地有结构的步骤。

