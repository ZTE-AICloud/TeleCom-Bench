import ast
import json
import re
from typing import Callable, Optional, Union

from opencompass.registry import TEXT_POSTPROCESSORS
from opencompass.utils.clean_jsonstr import clean_str_to_json


@TEXT_POSTPROCESSORS.register_module("first-capital")
def first_capital_postprocess(text: str) -> str:
    for t in text:
        if t.isupper():
            return t
    return ""


@TEXT_POSTPROCESSORS.register_module("specified-options")
def extract_specified_options(text: str) -> str:
    options = re.findall(r"[A-E]", text)
    return "".join(sorted(options))


@TEXT_POSTPROCESSORS.register_module("multiple-select")
def multiple_select_postprocess(text: str) -> str:
    ret = set([t for t in text if t.isupper()])
    return "".join(sorted(ret))


def general_eval_wrapper_postprocess(
    text: str, postprocess: Optional[Union[str, Callable]] = None, **kwargs
) -> str:
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


def extract_non_reasoning_content(text):
    """
    Remove content within <think>...</think> tags and retain only the content after </think>.
    """
    # Use regular expression to find the closing </think> tag and keep content after it
    result = re.split(r"</think>", text, maxsplit=1)
    if len(result) > 1:
        return result[1].strip()  # Return content after </think>
    return text  # If </think> is not found, return the original text


def process_latex(text: str):
    text = text.replace("dfrac", "frac")
    return text


def extract_boxed_content(text: str) -> str:
    start = text.rfind(r"boxed{")
    if start == -1:
        return ""

    i = start + len(r"boxed{")
    stack = 1
    content = []

    while i < len(text) and stack > 0:
        if text[i] == "\\" and i + 1 < len(text):
            content.append(text[i])
            content.append(text[i + 1])
            i += 2
            continue
        elif text[i] == "{":
            stack += 1
        elif text[i] == "}":
            stack -= 1
            if stack == 0:
                break
        if stack > 0:
            content.append(text[i])
        i += 1

    return "".join(content).strip()

def general_en_postprocess(text: str) -> str:
    truncated_text = re.split(r"[\n]", text, 1)[0]

    no_punctuation = re.sub(r"[^\w\s\-(){}<>\[\]]", " ", truncated_text)

    cleaned_text = re.sub(r"\s+", " ", no_punctuation).strip()

    return cleaned_text

def latex_last_en(text: str) -> str:
    matches = extract_boxed_content(text)
    if matches:
        return general_en_postprocess(matches)
    return ""


def latex_last_mcq(text: str) -> str:
    text = extract_non_reasoning_content(text)
    latex_matches = extract_boxed_content(text)
    if latex_matches:
        return extract_specified_options(latex_matches)
    phase = ["answer:", "answer is:", "final answer", "option"]
    pattern = "|".join(re.escape(p) for p in phase)
    parts = re.split(pattern, text.strip(), flags=re.IGNORECASE)
    last_part = parts[-1]
    ans_part = last_part.split(".", 1)[0]
    return extract_specified_options(ans_part)


def json_str(text: str):
    if not isinstance(text, str):
        text = str(text)

    replacements = {
        "\\n": "\n",
        "'": '"',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r'(?i)"(none|null)"', "null", text)
    text = re.sub(r"\s+", " ", text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        return None

    json_str = text[start : end + 1]

    try:
        parsed = ast.literal_eval(json_str)
        return json.dumps(parsed, ensure_ascii=False)
    except (ValueError, SyntaxError, json.JSONDecodeError):
        return json_str


def str2json(text):
    return clean_str_to_json(text)


def eoa_tag_postprocessor(text):
    text = extract_non_reasoning_content(text)
    pattern = re.compile(r'.*\[正确答案\](.*?)<eoa>(?![^<]*<eoa>)', re.DOTALL)
    match = pattern.search(text)
    if match:
        text = match.group(1).strip()
    text = multiple_select_postprocess(text)
    return text