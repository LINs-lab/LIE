# export HF_ENDPOINT=https://hf-mirror.com
import os
import json
import re
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
from transformers import AutoModelForCausalLM

from math_verify import parse, verify
from oat_math_grader import boxed_reward_fn as oat_evaluate

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"

# ---- 选择题答案抽取 ----
# 按优先级从高到低排列的抽取模式
MULTICHOICE_PATTERNS = [
    # \boxed{A}
    r"\\boxed\{\s*\(?([A-Ea-e])\)?\s*\}",
    # **ANSWER: D** / ANSWER: D / answer: A
    r"(?i)\b(?:ANSWER)\s*[:：]\s*\*{0,2}\s*\(?([A-Ea-e])\)?\s*\*{0,2}",
    # **Final Answer:** A / **Final Answer:** **A)** / Final Answer: B
    r"(?i)(?:Final\s+Answer)\s*[:：]\s*\*{0,2}\s*\(?([A-Ea-e])\)?\s*\*{0,2}",
    # The answer is A / the answer is (B) / the correct answer is C
    r"(?i)(?:the\s+)?(?:correct\s+)?answer\s+is\s*[:：]?\s*\(?([A-Ea-e])\)?",
    # option A is correct / option B
    r"(?i)option\s+\(?([A-Ea-e])\)?\s+is\s+correct",
    # choose A / select B
    r"(?i)(?:choose|select)\s+\(?([A-Ea-e])\)?",
    # (A) 或 A) 或 A. 出现在最后几行，作为最终答案
    r"(?:^|\n)\s*\*{0,2}\(?([A-Ea-e])\)[\.\s\*]",
]


def extract_multichoice_answer(text: str):
    """
    从模型输出中抽取选择题答案（A-E）。
    按优先级依次尝试多种格式，返回大写字母或 None。
    """
    if not text:
        return None
    for pattern in MULTICHOICE_PATTERNS:
        matches = re.findall(pattern, text)
        if matches:
            # 取最后一个匹配（模型可能在推理过程中提到多个选项，最终答案通常在最后）
            return matches[-1].upper()
    return None


def labeling_multichoice(response: str, golden_answer: str) -> bool:
    """
    判断选择题回答是否正确。
    golden_answer 应为单个字母（如 "A"）。
    """
    extracted = extract_multichoice_answer(response)
    if extracted is None:
        return False
    return extracted == golden_answer.strip().upper()


def timeout(timeout_seconds: int = 10):
    if os.name == "posix":
        import signal

        def decorator(func):
            def handler(signum, frame):
                raise TimeoutError("verify timed out!")

            def wrapper(*args, **kwargs):
                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout_seconds)
                try:
                    return func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            return wrapper
        return decorator


@timeout(timeout_seconds=10)
def labeling_responses(responses: list[str], golden_answer: str):
    predict_answers = list(map(parse, responses))
    golden_answers = list(
        map(parse, ["$" + golden_answer + "$"] * len(responses)))
    labels = list(map(verify, golden_answers, predict_answers))
    return labels


def make_conv_zero(question):
    question = question + \
        "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
    content = f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {question}. Assistant:"
    return content


def make_conv_zero_code(question):
    question = question + "\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end."
    content = f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {question}. Assistant:"
    return content


def make_conv_prime_sft(question, tokenizer):
    # for math problem
    content = question + \
        "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
    # for code problem
    # content = question + "\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end."
    msg = [
        {"role": "user", "content": content}
    ]
    chat = tokenizer.apply_chat_template(
        msg, tokenize=False, add_generation_prompt=True)
    return chat


def apply_qwen_math_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def qwen3_template(question: str, tokenizer):
    prompt = question + \
        "  Let's think step by step and output the final answer within \\boxed{}."
    msg = [
        {"role": "user", "content": prompt}
    ]
    prompt = tokenizer.apply_chat_template(
        msg, add_generation_prompt=True, tokenize=False)

    return prompt


def simplerl_template(question: str):
    return (
        '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
        + question
        + '\nPlease reason step by step, and put your final answer within\\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n'
    )


def l1_template(question: str, budget, tokenizer):
    prompt = question + " Let's think step by step and output the final answer within \\boxed{}." + \
        f" Think for {budget} tokens."
    msg = [
        {"role": "user", "content": prompt}
    ]
    prompt = tokenizer.apply_chat_template(
        msg, add_generation_prompt=True, tokenize=False)

    return prompt


def l1_ours_template(question: str, budget, tokenizer):

    system_prompt = 'Your task is to follow a systematic, thorough reasoning process before providing the final solution. This involves analyzing, summarizing, exploring, reassessing, and refining your thought process through multiple iterations. Structure your response into two sections: Thought and Solution. In the Thought section, present your reasoning using the format: "<think>\n {thoughts} </think>\n". Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas. After "</think>\n," in the Solution section, provide the final, logical, and accurate answer, clearly derived from the exploration in the Thought section. If applicable, include the answer in \\boxed{} for closed-form results like multiple choices or mathematical solutions.' + f" Think for {budget} tokens."

    msg = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    prompt = tokenizer.apply_chat_template(
        msg, add_generation_prompt=True, tokenize=False)

    return prompt


def luffy_template(question: str, tokenizer):

    system_prompt = 'Your task is to follow a systematic, thorough reasoning process before providing the final solution. This involves analyzing, summarizing, exploring, reassessing, and refining your thought process through multiple iterations. Structure your response into two sections: Thought and Solution. In the Thought section, present your reasoning using the format: "<think>\n {thoughts} </think>\n". Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas. After "</think>\n," in the Solution section, provide the final, logical, and accurate answer, clearly derived from the exploration in the Thought section. If applicable, include the answer in \\boxed{} for closed-form results like multiple choices or mathematical solutions.'

    msg = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    prompt = tokenizer.apply_chat_template(
        msg, add_generation_prompt=True, tokenize=False)

    return prompt


def main(input_file, output_file, model_path, debug=False, remove_system=True, template='own', temperature=0.6, top_p=1.0, max_tokens=8192, n=1, force_generate=True, add_think_before_answer=False, add_oat_evaluate=False, any_true=False, skip_scoring=False, output_eval=None, no_split_think=False, budget=1024, enigne="vllm", enable_thinking=None):

    df = pd.read_parquet(input_file)
    dec_output_path = output_file.replace('.jsonl', '') + '.decoded.jsonl'
    if force_generate or (not os.path.exists(dec_output_path)):
        # 数据处理
        messages = df['prompt'].tolist()

        if messages[0][0]['role'] == 'system':
            # assert remove_system is True
            if remove_system:
                print('remove system')
                assert messages[0][0]['role'] == 'system'
                messages = [message[1:] for message in messages]

            else:
                # assert remove_system is False
                print('not remove system')
        else:
            print('not remove system!!!')

        answers = df['reward_model'].tolist()
        # data_sources = df['data_source'].tolist()
        answers = [answer['ground_truth'] for answer in answers]

        # if debug:
        # answers = answers[:10]
        assert len(messages) == len(answers)

        print(messages[0])
        print(
            f"temperature: {temperature}, top_p: {top_p}, max_tokens: {max_tokens}, n: {n}")
        if enigne == 'hf':
            outputs = generate_hf(messages, model_path, template=template, temperature=temperature,
                                  top_p=top_p, max_tokens=max_tokens, n=n, budget=budget)
        elif enigne == 'vllm':
            outputs = generate_vllm(messages, model_path, template=template, temperature=temperature,
                                    top_p=top_p, max_tokens=max_tokens, n=n, budget=budget, enable_thinking=enable_thinking)
        # rets = {}

        # save the outputs first
        data_sources = [df['data_source'].iloc[i]
                        for i in range(len(df)) for j in range(n)]
        with open(dec_output_path, 'w') as fo:
            for i, output in enumerate(outputs):
                prompt = output.prompt
                for j in range(n):
                    generated_text = output.outputs[j].text
                    item = {
                        'prompt': prompt,
                        'generated_text': generated_text,
                        'answer': answers[i],
                        'data_source': df['data_source'].iloc[i]
                    }
                    fo.write(json.dumps(item) + '\n')

        # format sort prompts, outputs, answers
        assert len(outputs[0].outputs) == n
        prompts = [out.prompt for out in outputs for j in range(n)]
        answers = [answers[i] for i in range(len(outputs)) for j in range(n)]
        lengths = [len(out.outputs[j].token_ids)
                   for out in outputs for j in range(n)]
        outputs = [out.outputs[j].text for out in outputs for j in range(n)]

    else:
        print('Found already decoded file, skip decoding...')
        jss = []
        with open(dec_output_path, 'r') as f:
            for line in f:
                jss.append(json.loads(line))

        outputs = [item['generated_text'] for item in jss]
        prompts = [item['prompt'] for item in jss]
        answers = [item['answer'] for item in jss]
        # data_source: 优先从 decoded jsonl 读取，兼容旧格式时从 parquet 回退
        if 'data_source' in jss[0]:
            data_sources = [item['data_source'] for item in jss]
        else:
            print('警告: decoded jsonl 中无 data_source 字段，从 parquet 回退读取')
            data_sources = [df['data_source'].iloc[i // n]
                            for i in range(len(jss))]
        lengths = None  # 仅打分模式无 token 长度信息

    from collections import defaultdict
    rets = defaultdict(list)
    save_data = []
    avg = 0
    from tqdm import tqdm

    print('Scoring...')
    if skip_scoring:
        return

    # for i, output in tqdm(enumerate(outputs)):
    diff_cnt = 0
    for i in tqdm(range(len(outputs)), total=len(outputs)):
        # print(i)
        generated_text = outputs[i]
        prompt = prompts[i]
        answer = answers[i]
        think_format = False
        if prompt.endswith(THOUGHT_DELIMITER_START+'\n') or add_think_before_answer is True:
            generated_text = THOUGHT_DELIMITER_START + '\n' + generated_text
            think_format = True
        if no_split_think:
            think_format = False
        labels = None
        if think_format:
            try:
                generated_text = generated_text.split(THOUGHT_DELIMITER_END)[1]
            except Exception as e:
                labels = [False]

        if labels is None:
            try:
                labels = labeling_responses([generated_text,], answer)
            except Exception as e:
                labels = [False]

        # 如果 math_verify 判定为 False，再尝试选择题匹配（兼容 arc_c / gpqa / mmlu 等）
        if labels[0] is False or labels[0] == False:
            mc_correct = labeling_multichoice(generated_text, answer)
            if mc_correct:
                labels = [True]

        if add_oat_evaluate:
            new_label = oat_evaluate(generated_text, answer, fast=False)
            new_label = new_label[1] == 1.0
            if any_true is True:
                if labels[0] is False and new_label is True:
                    diff_cnt += 1
                    # breakpoint()
                labels = [labels[0] or new_label]
            else:
                labels = [new_label]

        rets[data_sources[i]].append(labels[0])

        save_data.append({
            'prompt': prompt,
            'generated_text': generated_text,
            'answer': answer,
            'correctness': labels[0],
            'data_source': data_sources[i]
        })
        if labels[0]:
            avg += 1

    print('accuracy: ', avg / len(outputs))
    print('diff_cnt: ', diff_cnt)

    accs = []
    for data_source, labels in rets.items():
        # print(data_source, len(labels))
        acc = np.array(labels).mean()
        print(f'{data_source}: {acc}')
        accs.append(acc)

    print('avg acc: ', np.array(accs).mean())
    if lengths is not None:
        print('max length:', max(np.array(lengths)))

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in save_data:
                f.write(json.dumps(item) + '\n')
    except Exception as e:
        print(f'Error: {e}')
        print(f'Output file: {output_file}')


def generate_hf(messages, model_path, template='own', temperature=0.6, top_p=0.95, max_tokens=8192, n=1, budget=1024, batch_size=32):

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    # You may want to use bfloat16 and/or move to GPU here
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    gen_prompts = []
    for i in range(len(messages)):
        cur_message = messages[i]
        prompt = tokenizer.apply_chat_template(
            cur_message,
            tokenize=False,
            add_generation_prompt=True,
            thinking_budget=budget,
            # return_tensors="pt"
        )
        gen_prompts.append(prompt)
        if i == 0:
            print('Example input: ', prompt)

    input_ids = tokenizer(gen_prompts, return_tensors="pt",
                          padding=True).input_ids
    print(input_ids.shape)

    outputs = []
    for i in range(0, len(input_ids), batch_size):
        batch_messages = input_ids[i: i+batch_size]
        outputs = model.generate(batch_messages.to(model.device), max_new_tokens=max_tokens,
                                 temperature=temperature,
                                 top_p=top_p,
                                 num_return_sequences=n
                                 )
        print(outputs)
        output_text = tokenizer.batch_decode(outputs)
        print(outputs)
        outputs.append(output_text)

    return outputs


def generate_vllm(messages, model_path, template='own', temperature=0.6, top_p=0.95, max_tokens=8192, n=1, budget=1024, top_k=-1, type='original', enable_thinking=None):
    # vllm模型加载
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # max_tokens is for the maximum length for generation.

    if type == 'original':
        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n, top_k=top_k)
    else:
        stop_token_ids = tokenizer.eos_token_id
        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n, stop_token_ids=stop_token_ids)

    print(torch.cuda.device_count())
    llm = LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(
    ), gpu_memory_utilization=0.85, max_model_len=max_tokens)  # 替换成本地路径

    gen_prompts = []

    for i in range(len(messages)):
        cur_message = messages[i]
        if template == 'own':
            if enable_thinking != None:
                gen_prompt = tokenizer.apply_chat_template(
                    cur_message,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking
                )
            else:
                gen_prompt = tokenizer.apply_chat_template(
                    cur_message,
                    tokenize=False,
                    add_generation_prompt=True
                )
        elif template == 'seed':
            gen_prompt = tokenizer.apply_chat_template(
                cur_message,
                tokenize=False,
                add_generation_prompt=True,
                thinking_budget=budget
            )
        elif template == 'luffy':
            gen_prompt = luffy_template(cur_message[-1]['content'], tokenizer)
        elif template == 'simplerl':
            gen_prompt = simplerl_template(cur_message[-1]['content'])
        elif template == 'qwen':
            gen_prompt = apply_qwen_math_template(cur_message[-1]['content'])
        elif template == 'qwen3':
            gen_prompt = qwen3_template(cur_message[-1]['content'], tokenizer)
        elif template == 'prime':
            gen_prompt = make_conv_zero(cur_message[-1]['content'])
        elif template == 'prime_sft':
            gen_prompt = make_conv_prime_sft(
                cur_message[-1]['content'], tokenizer)
        elif template == 'prime_code':
            gen_prompt = make_conv_zero_code(cur_message[-1]['content'])
        elif template == 'l1':
            gen_prompt = l1_template(
                cur_message[-1]['content'], budget=budget, tokenizer=tokenizer)
        elif template == 'l1-ours':
            gen_prompt = l1_ours_template(
                cur_message[-1]['content'], budget=budget, tokenizer=tokenizer)
        elif template == 'no':
            gen_prompt = cur_message[-1]['content']
        else:
            raise ValueError(f'Invalid template: {template}')
        gen_prompts.append(gen_prompt)
        if i == 0:
            print('Example input: ', gen_prompt)
    if type == 'original':
        print("original!!", sampling_params)

        outputs = llm.generate(gen_prompts, sampling_params)
    else:
        # 批量处理thinking和final answer生成
        ignore_str = "Wait"
        NUM_IGNORE = 1
        current_prompts = gen_prompts.copy()
        outputs = []
        for i, p in enumerate(current_prompts):
            out = ''
            o = llm.generate(
                p,
                sampling_params=sampling_params
            )
            ignore_str = "Wait"
            max_tokens_thinking_tmp = budget
            for i in range(NUM_IGNORE):  # Num of times to skip stop token
                max_tokens_thinking_tmp -= len(o[0].outputs[0].token_ids)
                if max_tokens_thinking_tmp > 0:
                    out += o[0].outputs[0].text + ignore_str
                    p += o[0].outputs[0].text + ignore_str
                    sampling_params = SamplingParams(
                        max_tokens=max_tokens_thinking_tmp,
                        min_tokens=1,
                        stop_token_ids=stop_token_ids,
                        skip_special_tokens=False,
                        temperature=temperature
                    )
                    o = llm.generate(
                        p,
                        sampling_params=sampling_params
                    )
            ### Final answer ###
            # You can also append "Final Answer:" here like we do for some evaluations to prevent the model from just continuing to reason in its answer when early exiting
            out += o[0].outputs[0].text
            sampling_params = SamplingParams(
                max_tokens=32768,
                min_tokens=0,
                stop_token_ids=stop_token_ids,
                skip_special_tokens=False,
                temperature=0.0,
            )
            o = llm.generate(
                p,
                sampling_params=sampling_params,
            )
            out += o[0].outputs[0].text
            # You will see that after the "Wait" in the reasoning trace it fixes its answer
            print("With budget forcing:")
            print(out)
            outputs.append(out)

    return outputs


if __name__ == "__main__":
    import fire
    fire.Fire(main)
