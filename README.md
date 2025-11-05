# Safety Reasoning Dataset

Repository accompanying Safety Through Reasoning: An Empirical Study of Reasoning Guardrail Models - EMNLP 2025. We will release the dataset by November 12, 2025 and model on Huggingface by end of the month.

## Dataset Details

### Dataset Description

The Safety Reasoning Guard Dataset contains reasoning traces generated from open source reasoning models to provide justifications for labels in two existing datasets released by NVIDIA: Nemotron Content Safety Dataset V2 and CantTalkAboutThis topic control dataset. The reasoning contains justifications for labels of either stand-alone user prompts engaging with an LLM or pairs of user prompts and LLM responses that are either safe/unsafe or on-topic/off-topic.

The reasoning traces are extracted from DeepSeek-R1, Qwen3-32B, and gpt-oss models using specific prompting strategies and filtering as discussed in the accompanying paper. The dataset includes two types of reasoning traces: full-length and reduced-length (efficient) reasoning traces for optimized inference performance.

- **Language(s) (NLP)**: English
- **License**: CC-BY-4.0

### Dataset Sources

- **Paper**: [Safety Through Reasoning: An Empirical Study of Reasoning Guardrail Models](https://arxiv.org/abs/2505.20087)
- **Nemotron Content Safety Dataset V2**: [https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0)
- **CantTalkAboutThis Topic Control Dataset**: [https://huggingface.co/datasets/nvidia/CantTalkAboutThis-Topic-Control-Dataset](https://huggingface.co/datasets/nvidia/CantTalkAboutThis-Topic-Control-Dataset)

## Uses

### Direct Use

This dataset is intended for training reasoning guard classifier models for LLM content safety and topic control. Broadly, the intended use cases are:

- Training reasoning-based guardrail models to detect unsafe content in LLM interactions
- Developing topic control mechanisms that maintain topical coherence in conversational AI
- Fine-tuning reasoning models for alignment tasks (e.g., Nemotron)
- Deploying reasoning guard classifiers in products like NeMo Guardrails
- Evaluating AI systems' ability to provide justified safety classifications

### Out-of-Scope Use

This dataset should not be used for harmful, unethical, or malicious purposes. This dataset should not be used for:

- Training models to generate harmful or inappropriate content
- Bypassing content moderation systems
- Creating adversarial examples to test system vulnerabilities without proper authorization
- Developing systems that enable unsafe content generation

## Reasoning Trace Generation

This dataset augments two existing NVIDIA datasets ([Nemotron Content Safety Dataset V2](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) and [CantTalkAboutThis topic control dataset](https://huggingface.co/datasets/nvidia/CantTalkAboutThis-Topic-Control-Dataset)) with reasoning traces from state-of-the-art reasoning models.

### Generator Models

Reasoning traces were generated using three open-source reasoning models:
- **DeepSeek-R1-0528**
- **Qwen3-32B**
- **gpt-oss-120b**

These models were selected for their strong reasoning capabilities and ability to provide detailed explanations for their classification decisions.

### Generation Process

The reasoning traces were generated using specific prompting strategies as described in the [accompanying paper](https://arxiv.org/abs/2505.20087). The generation process includes:

1. **Full-length reasoning traces**: Complete reasoning chains that explain the model's thought process in detail
2. **Efficient reasoning traces**: Reduced-length versions optimized for faster inference

### Quality Filtering

To ensure dataset quality and consistency, samples were filtered based on prediction accuracy. Only samples where the reasoning model's predicted labels matched the ground truth labels from the base datasets were retained. This filtering process resulted in:
- **28,000 samples** from Nemotron Content Safety Dataset V2 (out of 30,007 training samples)
- **8,712 samples** from CantTalkAboutThis topic control dataset (out of 1,080 dialogues with multiple turns)

For each sample, the dataset includes reasoning traces from all three generator models, along with a `prediction_gt_mismatch_{generator_model}` field indicating whether each model's prediction aligns with the ground truth.

## Dataset Structure

The dataset consists of two separate files, each corresponding to one of the base datasets with added reasoning traces:

### 1. Nemotron Content Safety Dataset V2 Safety Reasoning Dataset (`aegis_v2_efficient_reasoning.jsonl`)

This dataset contains 28,000 samples with reasoning traces for content safety classification. Each record includes the following fields:

#### Existing Fields from Nemotron Content Safety Dataset V2:

- **id**: The unique ID for the sample
- **reconstruction_id_if_redacted**: ID for reconstructed samples if content was redacted. The prompt will say "REDACTED" if an ID exists in this column for redacted samples from the Suicide Detection dataset. In this case, the ID specified matches the sample in the Suicide dataset, which needs to be separately downloaded.
- **prompt**: The first user turn
- **response**: The first assistant turn. `null` if prompt-only example. Sometimes can have an empty string `''` instead of a null, in which case the response label would be present.
- **prompt_label**: Binary safety label (safe/unsafe)
- **response_label**: Binary safety label (safe/unsafe). `null` if prompt-only example.
- **prompt_label_source**: Always human annotated
- **response_label_source**: Either human, llm_jury, or refusal_data_augmentation
- **violated_categories**: Comma-separated list of categories in order of frequency of annotations

#### New Fields Added for Reasoning Traces:

- **prompt_harm_label**: Binary harm classification for prompt (harmful/unharmful)
- **response_harm_label**: Binary harm classification for response (harmful/unharmful)
- **safety_reasoning_{generator_model}**: Full-length reasoning trace from the specified reasoning model, explaining the safety classification decision
- **efficient_safety_reasoning_trace_{generator_model}**: Reduced-length reasoning trace from the specified reasoning model for efficient inference
- **prediction_gt_mismatch_{generator_model}**: Boolean indicating whether the reasoning model's prediction matches the ground truth label

Note: `{generator_model}` can be one of: `deepseek_r1_0528`, `gpt_oss_120b`, or `qwen3_32b`

### 2. Topic Control Reasoning Dataset (CantTalkAboutThis Dataset)

This dataset contains 8,712 samples with reasoning traces for topic control classification. The dataset includes dialogues across nine domains: health, banking, travel, education, finance, insurance, legal, real estate, and computer troubleshooting. Each record includes the following fields:

#### Existing Fields from CantTalkAboutThis Dataset:

- **system_instruction**: The dialogue policy given to the model, usually a complex set of instructions on topics allowed and not allowed
- **conversation_history**: The full conversation history leading up to the current user message, including both the main topic and distractor turns (list of role and content pairs)
- **last_user_message**: The current user message to be classified for topic adherence
- **gt_topic_classification**: Ground truth classification (on-topic/off-topic)

#### New Fields Added for Reasoning Traces:

- **safety_reasoning_trace_{generator_model}**: Full-length reasoning trace from the specified reasoning model, explaining the topic classification decision
- **efficient_reasoning_trace_{generator_model}**: Reduced-length reasoning trace from the specified reasoning model for efficient inference
- **prediction_gt_mismatch_{generator_model}**: Boolean indicating whether the reasoning model's prediction matches the ground truth label

Note: `{generator_model}` can be one of: `deepseek_r1_0528`, `gpt_oss_120b`, or `qwen3_32b`

## Curation Rationale

The dataset addresses the need for explainable and reasoning-based guardrail models in AI safety. Traditional guardrail models provide classification decisions without justification, making it difficult to understand their decision-making process. By incorporating reasoning traces from state-of-the-art reasoning models, this dataset enables the development of guardrail systems that can:

1. Provide transparent explanations for safety classifications
2. Improve trust and interpretability in content moderation systems
3. Enable fine-tuning of reasoning models for alignment tasks
4. Support research into reasoning-based approaches for AI safety
5. Enable adaptability to custom safety policies

## Source Data

The dataset is created by augmenting two existing NVIDIA datasets with reasoning traces:

### Base Dataset 1: Nemotron Content Safety Dataset V2

**[Nemotron Content Safety Dataset V2](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0)** contains 33,416 annotated interactions between humans and LLMs, split into 30,007 training samples, 1,445 validation samples, and 1,964 test samples.

**Data Collection Method:**
- Hybrid: Human annotations with synthetic LLM augmentations

**Original Data Sources:**
- Human-written prompts collected from the Anthropic RLHF, Do-Anything-Now DAN, and AI-assisted Red-Teaming datasets
- LLM responses generated from Mistral-7B-v0.1
- Response safety labels generated using ensemble of 3 LLMs: Mixtral-8x22B-v0.1, Mistral-NeMo-12B-Instruct, and Gemma-2-27B-it
- Additional refusal data generated using Gemma-2-27B using custom prompt-engineered deflection strategies

**Labeling Method:**
- Human - Overall content safety labels for the conversation
- Synthetic - Response safety label augmentation where needed

**Annotation Process:**
Quality Assurance (QA) is maintained through regular review by project leads. Two to three times per week, leads randomly select fifteen questions of every one hundred completed by three annotators for reevaluation, accounting for at least fifteen percent of the data analyzed for three-way agreement. These corrections are sent to individual annotators as audits with explanations referencing project guidelines. Mandatory virtual group training sessions are held every two weeks to ensure consistent understanding.

**Annotators:**
Throughout the three-month timespan of the Content Moderation Guardrails project, twelve annotators on average were employed at any given time. Four annotators come from Engineering backgrounds specializing in data analysis and collection, gaming, and robotics. Eight annotators have backgrounds in Creative Writing, with specialization in linguistics, research and development, and other creative arts. All annotators reside in the United States, from various ethnic and religious backgrounds, providing representation across race, age, and social status. All annotators volunteered for this work after being informed of its potentially sensitive nature.

### Base Dataset 2: CantTalkAboutThis

**[CantTalkAboutThis](https://huggingface.co/datasets/nvidia/CantTalkAboutThis-Topic-Control-Dataset)** contains 1,080 dialogues with distractor turns across nine domains (health, banking, travel, education, finance, insurance, legal, real estate, and computer troubleshooting).

**Data Collection Method:**
- Synthetic

**Original Data Sources:**
The dataset was created using a pipeline to synthetically generate conversations and distractors, as described in the accompanying paper. This commercially-friendly version was generated using the Mixtral-8x7B-Instruct model. An evaluation dataset that is human annotated with more complex, realistic distractors is additionally provided for evaluating model performance.


## Personal and Sensitive Information

**Nemotron Content Safety Dataset V2 Safety Reasoning Dataset:**
The Nemotron Content Safety Dataset V2 portion has been carefully reviewed to remove personal information.

**Topic Control Reasoning Dataset (CantTalkAboutThis Dataset):**
The topic control portion does not contain any real-world personal or sensitive data.

## Bias, Risks, and Limitations

**Biases**: 
- The reasoning traces reflect the biases and reasoning patterns of the specific models used (DeepSeek-R1, Qwen3-32B, gpt-oss)
- The base datasets (Nemotron Content Safety Dataset V2 and CantTalkAboutThis) may contain inherent biases from their curation process

**Risks**: 
- Reasoning traces may not generalize to all types of safety violations or off-topic scenarios
- The reduced-length reasoning traces may lose important context or nuance compared to full-length traces
- Models trained on this data may inherit reasoning patterns that don't align with all safety frameworks

**Limitations**:
- The dataset is limited to English language content (may contain small samples in other languages in the Nemotron Content Safety Dataset V2 portion)
- Coverage is restricted to the safety categories and domains present in the base datasets
- The reasoning quality depends on the capabilities of the source reasoning models

**Legal Compliance**: 
Users of this data are responsible for ensuring its appropriate use. The dataset should not be utilized in manners that conflict with legal and ethical standards.


## Recommendations

Users should be made aware of the risks, biases, and limitations of the dataset. When using this dataset to train guardrail models:

- Validate model performance on diverse test sets beyond this dataset
- Consider combining with other safety datasets for broader coverage
- Monitor for potential biases in reasoning patterns
- Ensure human oversight for high-stakes safety decisions
- Regularly update models as reasoning capabilities improve

## Citation

**BibTeX:**

```
@inproceedings{sreedhar-etal-2025-safety,
    title = "Safety Through Reasoning: An Empirical Study of Reasoning Guardrail Models",
    author = "Sreedhar, Makesh Narsimhan  and
      Rebedea, Traian  and
      Parisien, Christopher",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.1193/",
    pages = "21862--21880",
    ISBN = "979-8-89176-335-7",
    abstract = "Reasoning-based language models have demonstrated strong performance across various domains, with the most notable gains seen in mathematical and coding tasks. Recent research has shown that reasoning also offers significant benefits for LLM safety and guardrail applications. In this work, we conduct a comprehensive analysis of training reasoning-based guardrail models for content moderation, with an emphasis on generalization to custom safety policies at inference time. Our study focuses on two key dimensions: data efficiency and inference efficiency. On the data front, we find that reasoning-based models exhibit strong sample efficiency, achieving competitive performance with significantly fewer training examples than their non-reasoning counterparts. This unlocks the potential to repurpose the remaining data for mining high-value, difficult samples that further enhance model performance. On the inference side, we evaluate practical trade-offs by introducing reasoning budgets, examining the impact of reasoning length on latency and accuracy, and exploring dual-mode training to allow runtime control over reasoning behavior. Our findings will provide practical insights for researchers and developers to effectively and efficiently train and deploy reasoning-based guardrails models in real-world systems."
}
```

## Dataset Card Authors

* Makesh Narsimhan Sreedhar
* Traian Rebedea
* Christopher Parisien

## Dataset Card Contact

* Makesh Narsimhan Sreedhar {makeshn@nvidia.com}
* Traian Rebedea {trebedea@nvidia.com}
