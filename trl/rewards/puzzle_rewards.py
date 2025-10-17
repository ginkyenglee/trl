JUDGE_SYSTEM = "You are a strict judge. Output ONLY a number from 0 to 10."

hallucination_prompt = """- Check for any fabricated or invented information not supported by the INPUT.
- Check for any fabricated or invented information not supported by the INPUT.
-"Hallucination" means any factual claim, entity, or detail that cannot be directly traced back to the INPUT.
- Paraphrasing or summarization of INPUT is acceptable and not hallucination.
- If additional information (facts, numbers, names, background knowledge) appears that is not in INPUT, treat it as hallucination.
- Do not infer, assume, or imagine anything that is not explicitly in the INPUT. Only use what is directly present in the INPUT.
- The INPUT is from speech recognition (ASR), so spoken-style errors or disfluencies should not be treated as hallucinations.
Scoring:
- 0: Entire Answer is hallucinated, no part is supported by INPUT.
- 1–3: Majority of Answer is hallucinated, only minor overlap with INPUT.
- 4–6: Main idea comes from INPUT but includes significant fabricated details.
- 7–9: Answer is mostly grounded in INPUT, with only minor unsupported additions.
- 10: Fully faithful, no hallucination, all details grounded in INPUT.

Return only the number representing the hallucination score.

#Prompts:
{system_prompt}

#INPUT:
{user}

#Answer:
{completion}

Return **only one integer from 0 to 10 (inclusive)** representing the hallucination score, with no additional text, symbols, or explanations."""

formatting_prompt = """- Evaluate whether the Answer follows the required format and structure as specified in the GUIDELINES properties provided in the INPUT.
- If [OUTPUT_FORMAT] is provided, it is treated as an OpenAI tools-style schema. The Answer must therefore be a dictionary consisting of the required properties defined in OUTPUT_FORMAT. Each property must be present, type-consistent, and semantically follow the description guidance.
- If the Prompt or Input requires JSON format as Answer, Answer MUST start with `{{` for the properties of OUTPUT_FORMAT if provided. If it starts with ```json (triple backticks + json) or {{"type":"object", ...}}, deduct points.

Scoring Guidelines for format_alignment:
- 0: Completely ignores the required format (e.g., free text, unrelated structure).
- 1–3: Severe misalignment. Some resemblance to JSON or required structure, but missing most required properties or entirely wrong wrapping (e.g., outer schema, code fences).
- 4–6: Partial alignment. Contains some required properties but others missing; or properties present but types wrong; or significant additionalProperties not allowed.
- 7–9: Mostly correct. Required properties are present and types mostly match. Minor format issues (e.g., extra whitespace, small ordering issues, or one property mistyped).
- 10: Perfect alignment. All required properties included, types correct, descriptions respected, no extra fields, starts exactly with `{{` and adheres strictly to OUTPUT_FORMAT.

#Prompts:
{system_prompt}

#INPUT:
{user}

#Answer:
{completion}

Return **only one integer from 0 to 10 (inclusive)** representing the format_alignment score, with no additional text, symbols, or explanations."""

summary_prompt = """- Evaluate whether the Answer follows the required format and structure as specified in the GUIDELINES properties provided in the INPUT.
- Evaluate whether all clinically relevant or clearly necessary details have been appropriately included.
- Evaluate whether the summary presents only the essential information (key events, critical details, main points) while removing unnecessary modifiers and integrating duplicate content.
- Assess how naturally the sentences flow and whether the structure facilitates easy understanding.

Considering all above criteria, assess the overall clinical usability and quality for summarization:
- 0: The summary is clinically unusable due to poor accuracy, incompleteness, hallucinations, format violations, or poor readability/conciseness.
- 10: The summary is clinically sound, fully usable, and meets high standards in accuracy, completeness, hallucination avoidance, format compliance, conciseness, and readability.

#Prompts:
{system_prompt}

#INPUT:
{user}

#Answer:
{completion}

Return **only one integer from 0 to 10 (inclusive)** representing the summarization score, with no additional text, symbols, or explanations."""

grammar_prompt = """- Evaluate whether all utterances from the original input are fully preserved without unnecessary changes to words or expressions, maintaining the original sentence form.
- Score 0: Unnecessary changes to words or expressions occur in every sentence, resulting in no preservation of the original expressions.
- Score 10: All utterances are fully preserved, with no unnecessary changes to words or expressions, and the original sentences and expressions are maintained exactly.
    Example:
    Original: [Clinician] Hello? This is your first time here.
    Output: [Clinician] Hello. This is your first visit. → Some expression changes, so points are deducted.
    If such expression changes occur in every sentence, assign a score of 0.

#Prompts:
{system_prompt}

#INPUT:
{user}

#Answer:
{completion}

Return **only one integer from 0 to 10 (inclusive)** representing the grammar score, with no additional text, symbols, or explanations."""


from openai import OpenAI
from openai import AsyncOpenAI
client = AsyncOpenAI()
#client = OpenAI()  # OPENAI_API_KEY 환경변수 사용

import asyncio

async def _reward_hallucination_with_openai_async(prompts, completions, task, **kwargs):
    results = [0.0] * len(prompts)
    reward_inference_models = kwargs['reward_inference_models']
    async def one(i, prompt, completion, task, reward_inference_models):
        try:
            system_part = prompt[0]['content'].split("##INPUT")[0]
            user_part   = prompt[0]['content'].split("##INPUT")[1]
            input_text  = hallucination_prompt.format(
                system_prompt=system_part,
                user=user_part,
                completion=completion[0]['content'],
            )
            resp = await client.responses.create(
                model=reward_inference_models,
                instructions=JUDGE_SYSTEM,
                input=input_text,
            )
            try:
                score = float(resp.output_text)
            except:
                print(f"[{i}] parse error: {resp.output_text}")
                score = 0.0
        except Exception as e:
            print(f"[{i}] exception: {e}")
            score = 0.0
        results[i] = score

    await asyncio.gather(*[
        one(i, p, c, t, r) for i, (p, c, t, r) in enumerate(zip(prompts, completions, task, reward_inference_models))
    ])
    return results


def reward_with_openai_judge_hallucination(prompts, completions, task, **kwargs):
    return asyncio.run(_reward_hallucination_with_openai_async(prompts, completions, task, **kwargs))


async def _reward_task_with_openai_async(prompts, completions, task,  **kwargs):
    results = [0.0] * len(prompts)
    reward_inference_models = kwargs['reward_inference_models']
    async def one(i, prompt, completion, task, reward_inference_models):
        try:
            system_part = prompt[0]['content'].split("##INPUT")[0]
            user_part   = prompt[0]['content'].split("##INPUT")[1]
            if task in ['grammar']:
                task_prompt = grammar_prompt
            elif task in ['concise','descriptive','grammar','structured_summarize']:
                task_prompt = summary_prompt
            else:
                task_prompt = formatting_prompt
            input_text  = task_prompt.format(
                system_prompt=system_part,
                user=user_part,
                completion=completion[0]['content'],
            )
            resp = await client.responses.create(
                model=reward_inference_models,
                instructions=JUDGE_SYSTEM,
                input=input_text,
            )
            try:
                score = float(resp.output_text)
            except:
                print(f"[{i}] parse error: {resp.output_text}")
                score = 0.0
        except Exception as e:
            print(f"[{i}] exception: {e}")
            score = 0.0
        results[i] = score

    await asyncio.gather(*[
        one(i, p, c, t, r) for i, (p, c, t,r) in enumerate(zip(prompts, completions, task, reward_inference_models))
    ])
    return results


def reward_with_openai_judge_task(prompts, completions, task, **kwargs):
    return asyncio.run(_reward_task_with_openai_async(prompts, completions, task, **kwargs))