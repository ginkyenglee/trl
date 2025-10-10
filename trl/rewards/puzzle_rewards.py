JUDGE_SYSTEM = "You are a strict judge. Output ONLY a number from 0 to 10."

hallucination_prompt = """- Check for any fabricated or invented information not supported by the INPUT.
- "Hallucination" means any factual claim, entity, or detail that cannot be directly traced back to the INPUT.
- Paraphrasing or summarization of INPUT is acceptable and not hallucination.
- If additional information (facts, numbers, names, background knowledge) appears that is not in INPUT, treat it as hallucination.

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

Return only the number representing the hallucination score."""

format_prompt = """- Evaluate whether the Answer follows the required format and structure as specified in the GUIDELINES properties provided in the INPUT.
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

Return only the number representing the format_alignment score."""

from openai import OpenAI
from openai import AsyncOpenAI
client = AsyncOpenAI()
#client = OpenAI()  # OPENAI_API_KEY 환경변수 사용

import asyncio

async def _reward_with_openai_async(prompts, completions, task, **kwargs):
    results = [0.0] * len(prompts)

    async def one(i, prompt, completion, task):
        try:
            system_part = prompt[0]['content'].split("##INPUT")[0]
            user_part   = prompt[0]['content'].split("##INPUT")[1]
            input_text  = hallucination_prompt.format(
                system_prompt=system_part,
                user=user_part,
                completion=completion[0]['content'],
            )
            resp = await client.responses.create(
                model="o4-mini",
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
        one(i, p, c, t) for i, (p, c, t) in enumerate(zip(prompts, completions, task))
    ])
    return results


def reward_with_openai_judge_hallucination(prompts, completions, task, **kwargs):
    return asyncio.run(_reward_with_openai_async(prompts, completions, task, **kwargs))


def reward_with_openai_judge_format(prompts, completions, task, **kwargs):
    rewards = []
    for prompt, completion, t in zip(prompts, completions, task):
        resp = client.responses.create(
            model="gpt-4o-mini",
            input= JUDGE_SYSTEM + "\n\n" +format_prompt.format(system_prompt = prompt[0]['content'].split("##INPUT")[0],
                                                                            user = prompt[0]['content'].split("##INPUT")[1] ,
                                                                            completion = completion[0]['content']),
                    
            temperature=0,
        )
        text = resp.output_text
        try:
            score = float(text)
        except Exception:
            print("reward_with_openai_judge_format EXCEPTION!!!!!!!!!!!!", score)
            score = 0.0
        rewards.append(score)
    return rewards