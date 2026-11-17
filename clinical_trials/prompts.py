from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from clinical_trials.data import (
    EXCLUSION_LABELS,
    INCLUSION_LABELS,
    CriterionJudgment,
    ReferenceAssessment,
    parse_criteria,
)
from clinical_trials.verifier import HARMONY, PLAIN, THINK, THINK_CLOSE, THINK_OPEN

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

INCLUSION_HEADER = "Inclusion criteria:"
EXCLUSION_HEADER = "Exclusion criteria:"
FINAL_LABEL_PREFIX = "Final label:"


def _label_set(labels: Sequence[str]) -> str:
    return "{" + ", ".join(f'"{label}"' for label in labels) + "}"


SYSTEM_PROMPT = (
    "You are a helpful assistant for clinical trial recruitment. Your task is to compare a given patient note "
    "and the inclusion and exclusion criteria of a clinical trial to determine the patient's eligibility at the "
    "criterion level and then for the whole trial.\n"
    "The factors that allow someone to participate in a clinical study are called inclusion criteria. The factors "
    "that disqualify someone from participating are called exclusion criteria. They are based on characteristics "
    "such as age, gender, the type and stage of a disease, previous treatment history, and other medical "
    "conditions.\n"
    "You should check the inclusion criteria one-by-one, then the exclusion criteria one-by-one, and output the "
    "following three elements for each criterion:\n"
    "\tElement 1. For each criterion, briefly generate your reasoning process: First, judge whether the criterion "
    "is not applicable (not very common), where the patient does not meet the premise of the criterion. Then, "
    "check if the patient note contains direct evidence. If so, judge whether the patient meets or does not meet "
    "the criterion. If there is no direct evidence, try to infer from existing evidence, and answer one question: "
    "If the criterion is true, is it possible that a good patient note will miss such information? If impossible, "
    "then you can assume that the criterion is not true. Otherwise, there is not enough information.\n"
    "\tElement 2. If there is relevant information, you must generate a list of relevant sentence IDs in the "
    "patient note. If there is no relevant information, you must annotate an empty list.\n"
    "\tElement 3. Classify the patient eligibility for this specific criterion: for an inclusion criterion, the "
    f"label must be chosen from {_label_set(INCLUSION_LABELS)}; for an exclusion criterion, the label must be "
    f'chosen from {_label_set(EXCLUSION_LABELS)}. "not applicable" should only be used for criteria that are not '
    'applicable to the patient. "not enough information" should be used where the patient note does not contain '
    'sufficient information for making the classification. Try to use as less "not enough information" as '
    "possible because if the note does not mention a medically important fact, you can assume that the fact is "
    'not true for the patient. "included" denotes that the patient meets the inclusion criterion, while "not '
    'included" means the reverse. "excluded" denotes that the patient meets the exclusion criterion and should be '
    'excluded in the trial, while "not excluded" means the reverse.\n'
    "Finally, classify the patient eligibility for the whole trial with a trial-level label:\n"
    "\t0: not relevant or not eligible, the patient does not meet the inclusion criteria.\n"
    "\t1: excluded, the patient meets the inclusion criteria but is excluded by an exclusion criterion.\n"
    "\t2: eligible, the patient meets the inclusion criteria and is not excluded by any exclusion criterion.\n"
    "You should output only the assessment, exactly formatted as follows, with one line per criterion numbered "
    "as in the trial:\n"
    f"{INCLUSION_HEADER}\n"
    "0. <reasoning> Evidence sentences: [<sentence IDs>]. Label: <label>\n"
    "...\n"
    f"{EXCLUSION_HEADER}\n"
    "0. <reasoning> Evidence sentences: [<sentence IDs>]. Label: <label>\n"
    "...\n"
    f"{FINAL_LABEL_PREFIX} <0|1|2>"
)


def _listing(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return ", ".join(str(item) for item in value)


def trial_description(trial: Mapping[str, Any]) -> str:
    text = f"Title: {trial.get('brief_title') or ''}\n"
    text += f"Target diseases: {_listing(trial.get('diseases_list'))}\n"
    text += f"Interventions: {_listing(trial.get('drugs_list'))}\n"
    text += f"Summary: {trial.get('brief_summary') or ''}\n"
    text += f"{INCLUSION_HEADER}\n {parse_criteria(str(trial.get('inclusion_criteria') or ''))}\n"
    text += f"{EXCLUSION_HEADER}\n {parse_criteria(str(trial.get('exclusion_criteria') or ''))}\n"
    return text


def user_message(patient_note: str, trial: Mapping[str, Any]) -> str:
    message = f"Here is the patient note, each sentence is led by a sentence_id:\n{patient_note}\n\n"
    message += f"Here is the clinical trial:\n{trial_description(trial)}\n\n"
    message += "Output in the required format:"
    return message


def build_messages(patient_note: str, trial: Mapping[str, Any], system_in_user: bool = False) -> list[dict[str, str]]:
    user = user_message(patient_note, trial)
    if system_in_user:
        return [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user}"}]
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def plain_prompt(messages: Sequence[Mapping[str, str]]) -> str:
    return "\n\n".join(message["content"] for message in messages) + "\n"


def _apply_chat_template(messages: Sequence[Mapping[str, str]], tokenizer: PreTrainedTokenizerBase) -> str:
    text = tokenizer.apply_chat_template([dict(m) for m in messages], tokenize=False, add_generation_prompt=True)
    return str(text)


def render_prompt(messages: Sequence[Mapping[str, str]], tokenizer: PreTrainedTokenizerBase) -> tuple[str, bool]:
    if getattr(tokenizer, "chat_template", None):
        return _apply_chat_template(messages, tokenizer), True
    return plain_prompt(messages), False


_FORMAT_PROBE = "ARIX response format probe"


def _chat_template_text(tokenizer: PreTrainedTokenizerBase) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, Mapping):
        return "\n".join(str(value) for value in template.values())
    return str(template or "")


def detect_response_format(tokenizer: PreTrainedTokenizerBase) -> str:
    template = _chat_template_text(tokenizer)
    if not template:
        return PLAIN
    if "<|channel|>" in template and "<|message|>" in template:
        return HARMONY
    try:
        rendered = _apply_chat_template([{"role": "user", "content": _FORMAT_PROBE}], tokenizer)
    except Exception:
        rendered = ""
    generation_prompt = rendered.rpartition(_FORMAT_PROBE)[2]
    if generation_prompt.rfind(THINK_OPEN) > generation_prompt.rfind(THINK_CLOSE):
        return THINK
    if THINK_CLOSE in generation_prompt:
        return PLAIN
    return THINK if THINK_CLOSE in template else PLAIN


def format_judgment(judgment: CriterionJudgment) -> str:
    reasoning = " ".join(judgment.reasoning.split())
    if reasoning and reasoning[-1] not in ".!?":
        reasoning += "."
    evidence = ", ".join(str(sentence_id) for sentence_id in judgment.sentence_ids)
    parts = [f"{judgment.index}.", reasoning, f"Evidence sentences: [{evidence}]. Label: {judgment.label}"]
    return " ".join(part for part in parts if part)


def format_assessment(assessment: ReferenceAssessment) -> str:
    lines = [
        INCLUSION_HEADER,
        *(format_judgment(judgment) for judgment in assessment.inclusion),
        EXCLUSION_HEADER,
        *(format_judgment(judgment) for judgment in assessment.exclusion),
        f"{FINAL_LABEL_PREFIX} {assessment.label}",
    ]
    return "\n".join(lines)
