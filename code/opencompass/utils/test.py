import reextract_non_reasoning_contentextract_non_reasoning_contentextract_non_reasoning_contentextract_non_reasoning_contentextract_non_reasoning_content
from typing import Callable, Optional, Union


def general_postprocess(text: str) -> str:
    # Cut off the first newline, period, or comma
    if not isinstance(text, str) or not text.strip():
        return ""
    truncated_text = re.split(r'[\n,.]', text, 1)[0]

    # Remove punctuation
    no_punctuation = re.sub(r'[^\w\s]', '', truncated_text)

    # Remove article
    no_articles = re.sub(r'\b(a|an|the)\b',
                         '',
                         no_punctuation,
                         flags=re.IGNORECASE)

    # Remove duplicated blank spaces
    cleaned_text = re.sub(r'\s+', ' ', no_articles).strip()
    return cleaned_text


# 删除空行、重复空格，保留字母、数字和特定符号
def general_en_postprocess(text: str) -> str:
    # Cut off the first newline, period, or comma
    truncated_text = re.split(r'[\n]', text, 1)[0]

    # Remove punctuation
    no_punctuation = re.sub(r'[^\w\s\-(){}<>\[\]]', ' ', truncated_text)

    # Remove duplicated blank spaces
    cleaned_text = re.sub(r'\s+', ' ', no_punctuation).strip()

    return cleaned_text


def general_cn_postprocess(text: str) -> str:
    truncated_text = re.split(r'[\n。]', text, 1)[0]

    no_punctuation = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', truncated_text)

    pre_text = re.sub(r'\s+', ' ', no_punctuation).strip()
    import jieba
    cleaned_text = ' '.join(jieba.cut(pre_text))
    return cleaned_text


def text2sql_remove_double_select(text: str) -> str:
    result_query = 'SELECT ' + text.replace('\n', ' ')
    result_query = result_query.replace('SELECT SELECT', 'SELECT')
    return result_query


# 返回第一个大写字母
def first_capital_postprocess(text: str) -> str:
    for t in text:
        if t.isupper():
            return t
    return ''


# 返回最后一个大写字母
def last_capital_postprocess(text: str) -> str:
    for t in text[::-1]:
        if t.isupper():
            return t
    return ''


# 返回第一个 A-D 的字母
def first_capital_postprocess_multi(text: str) -> str:
    match = re.search(r'([A-D]+)', text)
    if match:
        return match.group(1)
    return ''


# 返回所有 A-D 的字母，可选
def mulit_choice_postprocess(text: str) -> str:
    abcd_chars = [char for char in text if char in 'ABCD']
    abcd_chars = ''.join(abcd_chars)
    return abcd_chars


# 返回第一个匹配下面正则的内容
def first_option_postprocess(text: str, options=None, cushion=True) -> str:
    """Find first valid option for text."""

    # yapf: disable
    # flake8: noqa: W605
    if options is None:
        options = "ABCDabcd"
    patterns = [
        f'答案是?\s?([{options}])',
        f'答案是?\s?：([{options}])',
        f'答案是?\s?:([{options}])',
        f'答案应该?是\s?([{options}])',
        f'答案应该?选\s?([{options}])',
        f'答案为\s?([{options}])',
        f'答案选\s?([{options}])',
        f'选择?\s?([{options}])',
        f'故选?\s?([{options}])'
        f'答案：?\s?([{options}])',
        f'\\boxed\s*([{options}])',
        f'只有选?项?\s?([{options}])\s?是?对',
        f'只有选?项?\s?([{options}])\s?是?错',
        f'只有选?项?\s?([{options}])\s?不?正确',
        f'只有选?项?\s?([{options}])\s?错误',
        f'说法不?对选?项?的?是\s?([{options}])',
        f'说法不?正确选?项?的?是\s?([{options}])',
        f'说法错误选?项?的?是\s?([{options}])',
        f'([{options}])\s?是正确的',
        f'([{options}])\s?是正确答案',
        f'选项\s?([{options}])\s?正确',
        f'所以答\s?([{options}])',
        f'所以\s?([{options}][.。$]?$)',
        f'所有\s?([{options}][.。$]?$)',
        f'[\s，：:,]([{options}])[。，,\.]?$',
        f'[\s，,：:][故即]([{options}])[。\.]?$',
        f'[\s，,：:]因此([{options}])[。\.]?$',
        f'[是为。]\s?([{options}])[。\.]?$',
        f'因此\s?([{options}])[。\.]?$',
        f'显然\s?([{options}])[。\.]?$',
        f'答案是\s?(\S+)(?:。|$)',
        f'答案应该是\s?(\S+)(?:。|$)',
        f'答案为\s?(\S+)(?:。|$)',
        f'[Tt]he answer is \(?([{options}])\)?',
        f'[Tt]he answer is option \(?([{options}])\)?',
        f'[Tt]he correct answer is \(?([{options}])\)?',
        f'[Tt]he correct answer is option \(?([{options}])\)?',
        f'[Tt]he answer to the question is \(?([{options}])\)?',
        f'^选项\s?([{options}])',
        f'^([{options}])\s?选?项',
        f'(\s|^)[{options}][\s。，,：:\.$]',
        f'(\s|^)[{options}](\s|$)',
        f'1.\s?(.*?)$',
        f'1.\s?([{options}])[.。$]?$',
    ]
    cushion_patterns = [
        f'([{options}]):',
        f'[{options}]',
    ]
    # flake8: noqa
    # yapf: enable

    if cushion:
        patterns.extend(cushion_patterns)
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            outputs = match.group(0)
            for i in options:
                if i in outputs:
                    return i
    return ''


# 返回最后一个 options 设定的字母
def last_option_postprocess(text: str, options: str) -> str:
    match = re.findall(rf'([{options}])', text)
    if match:
        return match[-1]
    return ''


# 匹配数字，包括负数、小数
def first_number_postprocess(text: str) -> float:
    """Return the first number in a string."""
    # regex pattern to match numbers (both integers and decimals)
    pattern = r'(-?\d*\.?\d+)'

    # search the string for the pattern
    match = re.search(pattern, text)

    # if a match is found, return it. Otherwise, return None.
    return float(match.group(1)) if match else None


def extract_json_str(text: str) -> str:
    result = text.replace("\n", "").replace("`", "")
    # 定义正则表达式模式以匹配JSON字符串
    json_pattern = re.compile(r'\{.*\}')
    # 在输入字符串中搜索JSON字符串
    match = json_pattern.search(result)
    if match:
        result = match.group()
    return result


# 提取回答中的所有大写字母，直接拼接
def multiple_select_postprocess(text: str) -> str:
    ret = set([t for t in text if t.isupper()])
    return ''.join(sorted(ret))


def general_eval_wrapper_postprocess(text: str,
                                     postprocess: Optional[Union[
                                         str, Callable]] = None,
                                     **kwargs) -> str:
    """Wrapper for eval text repr. Especially for chatglmpro.

    Args:
        text(str): Text to be postprocessed.
        postprocess(Callable, optional): Original post processing function.
            Defaults to None.
        **kwargs: Other necessary kwargs for post processing function.
    """
    try:
        text = eval(text)
    except Exception:
        # in case empty input or other error, skip eval
        pass

    if postprocess:
        if isinstance(postprocess, str):
            postprocess = TEXT_POSTPROCESSORS.get(postprocess)
        return postprocess(text, **kwargs)
    else:
        return text


# 返回第一个匹配正则 answer_pattern 的文本
def match_answer_pattern(response_text: str, answer_pattern: str):
    match = re.search(answer_pattern, response_text)
    extracted_answer = match.group(1) if match else ''
    return extracted_answer


# 去除 cot 思考过程
def reason_remove_think_content(text):
    """
    Remove content within <think>...</think> tags and retain only the content after </think>.
    """
    # Use regular expression to find the closing </think> tag and keep content after it
    result = re.split(r'</think>', text, maxsplit=1)
    if len(result) > 1:
        return result[1].strip()  # Return content after </think>
    return text  # If </think> is not found, return the original text


def extract_boxed_content(text: str) -> str:
    pattern = r'\\boxed{'
    start = text.find(r'\boxed{')
    if start == -1:
        return ''
    i = start + len(r'\boxed{')
    stack = 1
    content = []
    while i < len(text):
        if text[i] == '{':
            stack += 1
        elif text[i] == '}':
            stack -= 1
            if stack == 0:
                break
        content.append(text[i])
        i += 1
    return ''.join(content).strip()


# 返回最后一个 boxed{} 里的内容，只保留 options 给定的字母
def reason_capital_postprocess(text: str, options=None) -> str:
    if options is None:
        options = "ABCDabcd"

    # 提取 \boxed{} 中的内容
    matches = re.findall(r'\\boxed{(.*?)}', text)
    if matches:
        content = matches[-1]
        # 找到 content 中第一个在 options 里的字符
        for char in content:
            if char in options:
                return char
    return first_option_postprocess(text)


# 返回最后一个 boxed{} 里的内容，只保留字母数字
def reason_capital_postprocess_multi(text: str) -> str:
    matches = re.findall(r'\\boxed{(.*?)}', text)
    if matches:
        return general_en_postprocess(matches[-1])
    return ''


# 返回最后一个 boxed{} 里的全部内容
def reason_option_postprocess(text: str, options=None, **kwargs) -> str:
    matches = re.findall(r'\\boxed{(.*?)}', text)
    if matches:
        return matches[-1]
    return ''


# 返回最后一个 boxed{} 里的全部大写字母 -> 分割，返回最后一个part出现的第一个大写字母
def reason_scq_postprocess(text: str) -> str:
    no_think_tag = reason_remove_think_content(text)
    latex_matches = re.findall(r'\\boxed{(.*?)}', no_think_tag)
    if latex_matches:
        return multiple_select_postprocess(latex_matches[-1])
    else:
        phase = ['answer is', 'final answer', 'answer:']
        pattern = re.compile('|'.join(re.escape(p) for p in phase), re.IGNORECASE)
        parts = pattern.split(no_think_tag)

        last_part = parts[-1] if parts else no_think_tag
        return first_capital_postprocess(last_part)


# 返回最后一个 boxed{} 里的全部大写字母 -> 分割，返回最后一个part出现的全部大写字母
def reason_mcq_postprocess(text: str) -> str:
    no_think_tag = reason_remove_think_content(text)
    latex_matches = re.findall(r'\\boxed{(.*?)}', no_think_tag)
    if latex_matches:
        return multiple_select_postprocess(latex_matches[-1])
    phase = ['answer:', 'answer is:', 'final answer', 'option']
    pattern = '|'.join(re.escape(p) for p in phase)
    parts = re.split(pattern, no_think_tag.strip(), flags=re.IGNORECASE)
    last_part = parts[-1]
    ans_part = last_part.split('.', 1)[0]
    return multiple_select_postprocess(ans_part)


# 返回最后一个 boxed{} 里的内容 -> 分割，返回最后一个part的内容
def reason_freeform_postprocess(text: str) -> str:
    no_think_tag = reason_remove_think_content(text)
    boxed = extract_boxed_content(no_think_tag)
    if boxed:
        return general_en_postprocess(boxed)
    phase = ['answer is', 'list is', 'answer:', 'list:']
    pattern = '|'.join(re.escape(p) for p in phase)
    parts = re.split(pattern, no_think_tag.strip(), flags=re.IGNORECASE)
    last_part = parts[-1]
    return general_en_postprocess(last_part)


def reason_math_postprocess(text: str) -> str:
    no_think_tag = reason_remove_think_content(text)
    boxed = extract_boxed_content(no_think_tag)
    if boxed:
        return general_en_postprocess(boxed)
    phase = ['answer:', 'answer is:', 'final answer', 'option']
    pattern = '|'.join(re.escape(p) for p in phase)
    parts = re.split(pattern, no_think_tag.strip(), flags=re.IGNORECASE)
    last_part = parts[-1]
    ans_part = last_part.split('.', 1)[0]
    return general_en_postprocess(ans_part)

text = """
        "prediction": "<think>\nOkay, so I need to figure out how to complete the sequence given the input < [ [ and make sure all the parentheses are closed properly. Let's start by breaking down what each character represents here. \n\nFirst, I notice there are three opening brackets here: a left angle bracket <, followed by two opening square brackets [ [. Each of these needs to be closed in the correct order. Wait, but wait, no—actually, parentheses usually have to be closed in the reverse order they were opened. So if you open a bracket, you have to close the most recent one first. But here, the input is < [ [ — so the order is < first, then two [.\n\nWait, but angle brackets and square brackets are different types. Usually, different types of brackets can be nested as long as each is properly closed. For example, in some contexts, you can have < [ ] >, which would be correct because the square brackets are closed before the angle brackets. But here, the input is < [ [ — so two square brackets after the angle bracket. So the sequence starts with <, then two [. So to close them properly, I need to close the most recent ones first. So the last opened is the second [, then the first [, and then the <.\n\nSo the current input is < [ [. Let's count them:\n\n1. <\n2. [\n3. [\n\nSo each opening bracket must be closed. The order of closing would be:\n\nFirst close the third character, which is [. Then the second [. Then the first <. So the correct closing sequence would be ] ] >. Wait, the angle bracket's closing is >, right? So putting that together, the complete sequence would be < [ [ ] ] >.\n\nBut wait, let me check again. Let's count each pair:\n\nThe first character is <, which needs a > to close.\n\nThen the next two are [ and [ — each needs a ] to close. Since they are opened in the order first [ and then another [, the closing should be in reverse. So the second [ is closed first with ], then the first [ with another ], then finally > for the <. So the complete string would be: < [ [ ] ] >.\n\nLet me visualize it:\n\n< [ [ ... ] ] >\n\nThe innermost [ is closed first, so ] comes next. Then another ] to close the first [. Then > to close the <. So after the initial < [ [, adding ] ] > would close them properly. So the full sequence would be < [ [ ] ] >.\n\nWait, but the input given is < [ [ — that's three characters. The output needs to complete the sequence properly. So the user is asking to add the necessary closing brackets. So the output should be the entire correct sequence, starting with the input and appending the necessary closing brackets.\n\nTherefore, the input is < [ [ — three characters. We need to add the closing ones. So the next three characters should be ] ] >. So putting it all together, the full sequence is < [ [ ] ] >.\n\nWait, but let me think again. Each opening bracket must be closed in the correct order. Let's check the counts. The angle bracket < is first, then two square brackets. So when closing, the last opened square bracket is closed first, then the previous square bracket, then the angle bracket. So yes, adding two ] and a >. So the answer is ] ] >.\n\nTherefore, the completed sequence would be < [ [ ] ] >. So the user's input is the first three characters, and the answer is the next three: ] ] >.\n</think>\n\nTo complete the sequence `< [ [`, we need to properly close each opening bracket in reverse order. The input starts with `<`, followed by two `[`. Proper closure requires closing the most recently opened brackets first: the second `[`, then the first `[`, and finally the `<`. This results in appending `] ] >`.\n\n**Step-by-step:**\n1. **Close the second `[`:** Add `]` → `< [ [ ]`.\n2. **Close the first `[`:** Add `]` → `< [ [ ] ]`.\n3. **Close the `<`:** Add `>` → `< [ [ ] ] >`.\n\nSo the answer is \\boxed{] ] >}.",
"""

print(reason_freeform_postprocess(text))
