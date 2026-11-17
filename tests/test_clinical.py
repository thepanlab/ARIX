from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from tiny_model import train_tokenizer
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

import arix.evaluation
import clinical_trials.data
from arix.diversity import self_bleu
from arix.task import Example, ResponseDecoder, Task, decode_for_verification
from clinical_trials import evaluate
from clinical_trials.analysis.trial_overlap import main as overlap_main
from clinical_trials.analysis.trial_overlap import trial_overlap
from clinical_trials.data import (
    CONSENT_SENTENCE,
    ClinicalDataConfig,
    ClinicalPair,
    CriterionJudgment,
    Qrel,
    ReferenceAssessment,
    attach_references,
    derive_trial_label,
    label_counts,
    load_corpus_pairs,
    load_corpus_trials,
    load_evaluation_pairs,
    load_matching_results,
    load_qrels,
    load_retrieved_trials,
    load_train_validation_pairs,
    load_training_pairs,
    number_patient_note,
    parse_criteria,
    parse_matching_output,
    regex_split_sentences,
    sentence_splitter,
    split_sentences,
    split_train_validation,
)
from clinical_trials.prompts import (
    SYSTEM_PROMPT,
    build_messages,
    detect_response_format,
    format_assessment,
    format_judgment,
    plain_prompt,
    render_prompt,
    trial_description,
    user_message,
)
from clinical_trials.task import ClinicalTask, build_example, build_examples, decode_responses
from clinical_trials.train import load_run_config
from clinical_trials.verifier import (
    HARMONY,
    PLAIN,
    THINK,
    binary_counts,
    binary_metrics,
    confusion_matrix,
    final_answer,
    is_correct,
    parse_final_label,
)

FIXTURES = Path(__file__).parent / "fixtures" / "clinical"
MATCHING = FIXTURES / "matching_results_trec_2021_SYNTHETIC.json"
CONFIGS = Path(__file__).parents[1] / "clinical_trials" / "configs"
QRELS = [
    ("trec-20211", "NCT02777398", 0),
    ("trec-20211", "NCT03727841", 0),
    ("trec-20211", "NCT00841555", 1),
    ("trec-20211", "NCT00430911", 1),
    ("trec-20211", "NCT04211974", 2),
    ("trec-20211", "NCT00003470", 2),
]
CHAT_TEMPLATE = (
    "{% for m in messages %}<s>{{ m['role'] }}\n{{ m['content'] }}</s>\n{% endfor %}"
    "{% if add_generation_prompt %}<s>assistant\n{% endif %}"
)

R1_BOS, R1_EOS = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"
R1_DISTILL_TEMPLATE = (
    "{% set ns = namespace(system_prompt='') %}"
    "{% for message in messages %}{% if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}"
    "{% endif %}{% endfor %}{{ bos_token }}{{ ns.system_prompt }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}{{ '<｜User｜>' + message['content'] }}{% endif %}"
    "{% if message['role'] == 'assistant' %}{% set content = message['content'] %}"
    "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"
    "{{ '<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>' }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<｜Assistant｜><think>\\n' }}{% endif %}"
)
R1_DISTILL_TEMPLATE_WITHOUT_PREFILL = R1_DISTILL_TEMPLATE.replace("<｜Assistant｜><think>\\n", "<｜Assistant｜>")
CLOSED_THINK_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n\n</think>\n\n{% endif %}"
)
HARMONY_TEMPLATE = (
    "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nReasoning: medium\n\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}<|start|>developer<|message|># Instructions\n\n{{ message['content'] }}<|end|>"
    "{% elif message['role'] == 'user' %}<|start|>user<|message|>{{ message['content'] }}<|end|>"
    "{% else %}<|start|>assistant<|channel|>final<|message|>{{ message['content'] }}<|end|>{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|start|>assistant{% endif %}"
)
HARMONY_TOKENS = ("<|start|>", "<|end|>", "<|message|>", "<|channel|>", "<|constrain|>", "<|return|>", "<|call|>")
ANSWER = "Inclusion criteria:\n0. Glioblastoma confirmed. Evidence sentences: [0]. Label: included\nExclusion criteria:"
R1_RESPONSES = [
    ("Okay, I check each criterion and finish with Final label: 2 unless an exclusion applies. Inclusion 0:", False),
    (f"Draft: Final label: 2.\n</think>\n\n{ANSWER}", True),
    (f"Exclusion 0 might apply, so Final label: 1? No, it does not.\n</think>\n\n{ANSWER}\nFinal label: 2", True),
]
HARMONY_RESPONSES = [
    ("<|channel|>analysis<|message|>We need criteria lines then Final label: 2. Let us check exclusion 4:", False),
    (
        f"<|channel|>analysis<|message|>Final label: 2 fits.<|end|><|start|>assistant<|channel|>final<|message|>"
        f"{ANSWER}",
        False,
    ),
    (
        "<|channel|>analysis<|message|>Exclusion 0 might apply, so Final label: 1? No.<|end|>"
        f"<|start|>assistant<|channel|>final<|message|>{ANSWER}\nFinal label: 2",
        True,
    ),
]


def trialgpt_parse_criteria(criteria: str) -> str:
    output = ""
    criteria = criteria.split("\n\n")
    idx = 0
    for criterion in criteria:
        criterion = criterion.strip()
        if "inclusion criteria" in criterion.lower() or "exclusion criteria" in criterion.lower():
            continue
        if len(criterion) < 5:
            continue
        output += f"{idx}. {criterion}\n"
        idx += 1
    return output


def trialgpt_print_trial(trial_info: dict, inc_exc: str) -> str:
    trial = f"Title: {trial_info['brief_title']}\n"
    trial += f"Target diseases: {', '.join(trial_info['diseases_list'])}\n"
    trial += f"Interventions: {', '.join(trial_info['drugs_list'])}\n"
    trial += f"Summary: {trial_info['brief_summary']}\n"
    if inc_exc == "inclusion":
        trial += "Inclusion criteria:\n %s\n" % trialgpt_parse_criteria(trial_info["inclusion_criteria"])  # noqa: UP031
    elif inc_exc == "exclusion":
        trial += "Exclusion criteria:\n %s\n" % trialgpt_parse_criteria(trial_info["exclusion_criteria"])  # noqa: UP031
    return trial


def trialgpt_numbered_note(patient: str) -> str:
    sents = split_sentences(patient)
    sents.append(
        "The patient will provide informed consent, and will comply with the trial protocol "
        "without any practical issues."
    )
    sents = [f"{idx}. {sent}" for idx, sent in enumerate(sents)]
    return "\n".join(sents)


@pytest.fixture(scope="module")
def pairs() -> list[ClinicalPair]:
    return load_corpus_pairs(FIXTURES, "trec_2021")[0]


@pytest.fixture(scope="module")
def matching() -> dict:
    return load_matching_results(MATCHING, "trec_2021")


def _pair(pairs: list[ClinicalPair], trial_id: str) -> ClinicalPair:
    return next(pair for pair in pairs if pair.trial_id == trial_id)


def _copy_corpus(tmp_path: Path, *keep: str) -> Path:
    target = tmp_path / "dataset" / "trec_2021"
    (target / "qrels").mkdir(parents=True)
    shutil.copy(FIXTURES / "trec_2021" / "qrels" / "test.tsv", target / "qrels" / "test.tsv")
    for name in keep:
        shutil.copy(FIXTURES / "trec_2021" / name, target / name)
    return tmp_path / "dataset"


def _tiny_tokenizer(pairs: list[ClinicalPair], chat_template: str | None) -> PreTrainedTokenizerFast:
    texts = [plain_prompt(build_messages(pair.patient_note, pair.trial)) for pair in pairs]
    backend = Tokenizer(models.WordLevel(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.train_from_iterator(texts, trainers.WordLevelTrainer(special_tokens=["<unk>", "<pad>", "<s>", "</s>"]))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", bos_token="<s>", eos_token="</s>"
    )
    if chat_template is not None:
        tokenizer.chat_template = chat_template
    return tokenizer


def _response_tokenizer(
    special: tuple[str, ...], plain: tuple[str, ...], chat_template: str
) -> PreTrainedTokenizerFast:
    tokenizer = train_tokenizer([text for text, _ in R1_RESPONSES + HARMONY_RESPONSES])
    tokenizer.add_tokens([AddedToken(token, special=True, normalized=False) for token in special])
    tokenizer.add_tokens([AddedToken(token, special=False, normalized=False) for token in plain])
    tokenizer.chat_template = chat_template
    return tokenizer


def _r1_distill_tokenizer(think_is_special: bool) -> PreTrainedTokenizerFast:
    think = ("<think>", "</think>")
    tokenizer = _response_tokenizer(
        (R1_BOS, R1_EOS, *(think if think_is_special else ())),
        ("<｜User｜>", "<｜Assistant｜>", *(() if think_is_special else think)),
        R1_DISTILL_TEMPLATE,
    )
    tokenizer.bos_token, tokenizer.eos_token = R1_BOS, R1_EOS
    return tokenizer


def _harmony_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = _response_tokenizer(HARMONY_TOKENS, (), HARMONY_TEMPLATE)
    tokenizer.eos_token = "<|return|>"
    return tokenizer


def _generated_ids(tokenizer: PreTrainedTokenizerFast, responses: list[tuple[str, bool]]) -> list[list[int]]:
    return [
        tokenizer.encode(text, add_special_tokens=False) + ([tokenizer.eos_token_id] if finished else [])
        for text, finished in responses
    ]


def test_fixture_pairs_load(pairs: list[ClinicalPair]) -> None:
    loaded, stats = load_corpus_pairs(FIXTURES, "trec_2021")
    assert [(p.patient_id, p.trial_id, p.label) for p in loaded] == QRELS
    assert loaded[0].uid == "trec_2021/trec-20211/NCT02777398"
    assert stats == {
        "qrels": 6,
        "qrels_labels": {"0": 2, "1": 2, "2": 2},
        "repeated_qrels": 0,
        "missing_trial": 0,
        "missing_patient": 0,
        "pairs": 6,
        "patients": 1,
        "trials": 6,
        "labels": {"0": 2, "1": 2, "2": 2},
    }
    assert len({id(p.patient_note) for p in pairs}) == 1
    assert all(p.trial["brief_title"] for p in pairs)


def test_trial_records_from_either_file(tmp_path: Path) -> None:
    from_corpus = load_corpus_trials(FIXTURES / "trec_2021" / "corpus.jsonl")
    from_retrieved = load_retrieved_trials(FIXTURES / "trec_2021" / "retrieved_trials.json")
    assert sorted(from_corpus) == sorted(from_retrieved) == sorted(trial for _, trial, _ in QRELS)
    for trial_id, info in from_corpus.items():
        assert {k: v for k, v in from_retrieved[trial_id].items() if k != "NCTID"} == info
        assert from_retrieved[trial_id]["NCTID"] == trial_id
    assert sorted(load_corpus_trials(FIXTURES / "trec_2021" / "corpus.jsonl", ["NCT00003470"])) == ["NCT00003470"]
    for files in (("queries.jsonl", "corpus.jsonl"), ("queries.jsonl", "retrieved_trials.json")):
        dataset = _copy_corpus(tmp_path / Path(files[1]).stem, *files)
        loaded, _ = load_corpus_pairs(dataset, "trec_2021")
        assert [(p.patient_id, p.trial_id, p.label) for p in loaded] == QRELS
    with pytest.raises(FileNotFoundError):
        load_corpus_pairs(_copy_corpus(tmp_path / "none", "queries.jsonl"), "trec_2021")


def test_qrels_parsing(tmp_path: Path) -> None:
    assert load_qrels(FIXTURES / "trec_2021" / "qrels" / "test.tsv") == [Qrel(*row) for row in QRELS]
    no_header = tmp_path / "no_header.tsv"
    no_header.write_text("p1\tNCT1\t2\np1\tNCT2\t0\n", encoding="utf-8")
    assert load_qrels(no_header) == [Qrel("p1", "NCT1", 2), Qrel("p1", "NCT2", 0)]
    for content in ("query-id\tcorpus-id\tscore\np1\tNCT1\t3\n", "p1\tNCT1\n", "p1\tNCT1\tx\n"):
        bad = tmp_path / "bad.tsv"
        bad.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError):
            load_qrels(bad)


def test_repeated_and_missing_judgments(tmp_path: Path) -> None:
    dataset = _copy_corpus(tmp_path, "queries.jsonl", "corpus.jsonl")
    qrels = dataset / "trec_2021" / "qrels" / "test.tsv"
    original = qrels.read_text(encoding="utf-8").rstrip("\n")
    extra = [
        "trec-20211\tNCT00003470\t2",
        "trec-20211\tNCT99999999\t1",
        "trec-29999\tNCT00003470\t0",
    ]
    qrels.write_text("\n".join([original, *extra]) + "\n", encoding="utf-8")
    loaded, stats = load_corpus_pairs(dataset, "trec_2021")
    assert len(loaded) == 6
    assert (stats["repeated_qrels"], stats["missing_trial"], stats["missing_patient"]) == (1, 1, 1)
    assert stats["qrels"] == 9 and stats["qrels_labels"] == {"0": 3, "1": 3, "2": 3}
    assert stats["labels"] == {"0": 2, "1": 2, "2": 2}
    with open(qrels, "a", encoding="utf-8") as handle:
        handle.write("trec-20211\tNCT00003470\t1\n")
    with pytest.raises(ValueError, match="conflicting"):
        load_corpus_pairs(dataset, "trec_2021")


def test_patient_note_numbered_like_trialgpt(pairs: list[ClinicalPair]) -> None:
    patient = json.loads((FIXTURES / "trec_2021" / "queries.jsonl").read_text(encoding="utf-8").splitlines()[0])
    note = number_patient_note(patient["text"])
    assert note == trialgpt_numbered_note(patient["text"]) == pairs[0].patient_note
    lines = note.split("\n")
    assert len(lines) == len(split_sentences(patient["text"])) + 1
    assert lines[0].startswith("0. Patient is a 45-year-old man with a history of anaplastic astrocytoma")
    assert lines[-1] == f"{len(lines) - 1}. {CONSENT_SENTENCE}"


def test_regex_sentence_fallback() -> None:
    text = (
        "Dr. Smith saw the pt. today. EF was 25%. He was started on Lasix (e.g. 40 mg) daily!\n"
        "Is he better? He reports\nno dyspnea at rest."
    )
    assert regex_split_sentences(text) == [
        "Dr. Smith saw the pt.",
        "today.",
        "EF was 25%.",
        "He was started on Lasix (e.g. 40 mg) daily!",
        "Is he better?",
        "He reports\nno dyspnea at rest.",
    ]
    assert regex_split_sentences("  ") == []
    assert regex_split_sentences('He said "stop." Then left') == ['He said "stop."', "Then left"]
    assert regex_split_sentences("He takes vitamin D. He feels well.") == ["He takes vitamin D.", "He feels well."]
    assert regex_split_sentences("He takes vitamin D. daily.") == ["He takes vitamin D. daily."]


@pytest.mark.parametrize(
    ("text", "sentences"),
    [
        (
            "Recent antibx use in the last month on prior admission. Was also txd for Cdiff at that time for 14 d. "
            "course with po vanco. Pt was initially admitted to the ICU",
            [
                "Recent antibx use in the last month on prior admission.",
                "Was also txd for Cdiff at that time for 14 d. course with po vanco.",
                "Pt was initially admitted to the ICU",
            ],
        ),
        (
            "Peripheral blood smear showed a moderate number of schistocytes and helmet cells.  Shiga-like "
            "toxin-producing E. coli (STEC) stx1/stx2 were found in stools. \nHe has no other underlying disease",
            [
                "Peripheral blood smear showed a moderate number of schistocytes and helmet cells.",
                "Shiga-like toxin-producing E. coli (STEC) stx1/stx2 were found in stools.",
                "He has no other underlying disease",
            ],
        ),
        (
            "HR is 110 and he looks ill with dry mucosa. V. cholerae was seen in dark-field microscopy of a fresh "
            "stool specimen. The lab study is as bellow:",
            [
                "HR is 110 and he looks ill with dry mucosa.",
                "V. cholerae was seen in dark-field microscopy of a fresh stool specimen.",
                "The lab study is as bellow:",
            ],
        ),
    ],
)
def test_regex_splitter_keeps_an_initial_before_a_lower_case_word(text: str, sentences: list[str]) -> None:
    assert regex_split_sentences(text) == sentences


def test_the_sentence_splitter_is_recorded_in_the_data_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = ClinicalDataConfig.from_dict({"dataset_dir": str(FIXTURES), "train_corpora": "trec_2021", "test_corpora": []})
    assert sentence_splitter() in ("punkt", "regex")
    for splitter, found in (("regex", None), ("punkt", regex_split_sentences)):
        monkeypatch.setattr(clinical_trials.data, "_punkt_sentence_tokenizer", lambda found=found: found)
        assert sentence_splitter() == splitter
        assert load_train_validation_pairs(cfg)[2]["sentence_splitter"] == splitter


def test_parse_criteria_like_trialgpt(pairs: list[ClinicalPair]) -> None:
    for pair in pairs:
        for section in ("inclusion_criteria", "exclusion_criteria"):
            assert parse_criteria(pair.trial[section]) == trialgpt_parse_criteria(pair.trial[section])
    text = "Inclusion Criteria:\n\n  Age >= 18 years  \n\nN/A\n\nSigned consent\n\n- EXCLUSION CRITERIA: none\n\n"
    assert parse_criteria(text) == trialgpt_parse_criteria(text) == "0. Age >= 18 years\n1. Signed consent\n"


def test_prompt_contains_numbered_sentences_and_criteria(pairs: list[ClinicalPair]) -> None:
    pair = _pair(pairs, "NCT00430911")
    system, user = build_messages(pair.patient_note, pair.trial)
    assert system == {"role": "system", "content": SYSTEM_PROMPT}
    assert user["role"] == "user"
    text = user["content"]
    assert text.startswith("Here is the patient note, each sentence is led by a sentence_id:\n0. Patient is a 45-year")
    assert f"\n{pair.patient_note.count(chr(10))}. {CONSENT_SENTENCE}\n" in text
    assert "Inclusion criteria:\n 0. Glioblastoma and anaplastic astrocytoma, supratentorial" in text
    assert "\n5. Informed consent\n" in text
    assert "Exclusion criteria:\n 0. Brain radiotherapy\n1. Chemotherapy\n2. Severe systemic disease\n" in text
    assert text.endswith("Output in the required format:")
    for pair in pairs:
        expected = trialgpt_print_trial(dict(pair.trial), "inclusion")
        expected += f"Exclusion criteria:\n {trialgpt_parse_criteria(pair.trial['exclusion_criteria'])}\n"
        assert trial_description(pair.trial) == expected
    for fragment in (
        '{"not applicable", "not enough information", "included", "not included"}',
        '{"not applicable", "not enough information", "excluded", "not excluded"}',
        "Element 2. If there is relevant information, you must generate a list of relevant sentence IDs",
        "0: not relevant or not eligible, the patient does not meet the inclusion criteria.",
        "1: excluded, the patient meets the inclusion criteria but is excluded by an exclusion criterion.",
        "2: eligible",
        "Inclusion criteria:\n0. <reasoning> Evidence sentences: [<sentence IDs>]. Label: <label>\n...\nExclusion",
        "Final label: <0|1|2>",
    ):
        assert fragment in SYSTEM_PROMPT


def test_prompt_rendering_with_and_without_chat_template(pairs: list[ClinicalPair]) -> None:
    pair = pairs[0]
    messages = build_messages(pair.patient_note, pair.trial)
    templated = _tiny_tokenizer(pairs, CHAT_TEMPLATE)
    text, is_templated = render_prompt(messages, templated)
    assert is_templated and text.startswith("<s>system\nYou are a helpful assistant")
    assert text.endswith("</s>\n<s>assistant\n")
    example = build_example(pair, templated)
    assert example.prompt == text and example.meta["add_special_tokens"] is False
    assert templated(example.prompt, add_special_tokens=False).input_ids[0] == templated.bos_token_id
    plain = _tiny_tokenizer(pairs, None)
    text, is_templated = render_prompt(messages, plain)
    assert not is_templated and text == f"{messages[0]['content']}\n\n{messages[1]['content']}\n"
    example = build_example(pair, plain)
    assert example.meta == {
        "add_special_tokens": True,
        "corpus": "trec_2021",
        "patient_id": "trec-20211",
        "trial_id": "NCT02777398",
    }
    assert (example.uid, example.target, example.reference) == (pair.uid, 0, "")


def test_references_from_synthetic_matching_file(pairs: list[ClinicalPair], matching: dict) -> None:
    kept, stats = attach_references(pairs, matching)
    every_label = {"0": 2, "1": 2, "2": 2}
    assert stats.as_dict() == {
        "pairs": 6,
        "kept": 6,
        "no_output": 0,
        "unusable_output": 0,
        "label_disagreement": 0,
        "dropped": 0,
        "labels": {"pairs": every_label, "with_output": every_label, "kept": every_label},
    }
    assert [p.uid for p in kept] == [p.uid for p in pairs]
    for pair in kept:
        assert pair.reference is not None and pair.reference.label == pair.label
        output = matching[pair.uid]
        assert [j.index for j in pair.reference.inclusion] == list(range(len(output["inclusion"])))
        assert [j.index for j in pair.reference.exclusion] == list(range(len(output["exclusion"])))
        text = format_assessment(pair.reference)
        assert parse_final_label(text) == pair.label and is_correct(text, pair.label)
        lines = text.split("\n")
        assert lines[0] == "Inclusion criteria:" and lines[-1] == f"Final label: {pair.label}"
        assert lines.index("Exclusion criteria:") == len(pair.reference.inclusion) + 1
    excluded = format_assessment(_pair(kept, "NCT00430911").reference)
    assert excluded == "\n".join(
        [
            "Inclusion criteria:",
            *(
                f"{i}. SYNTHETIC reasoning for inclusion criterion {i}. Evidence sentences: [0]. Label: included"
                for i in range(6)
            ),
            "Exclusion criteria:",
            "0. SYNTHETIC reasoning for exclusion criterion 0. Evidence sentences: []. Label: excluded",
            "1. SYNTHETIC reasoning for exclusion criterion 1. Evidence sentences: []. Label: not excluded",
            "2. SYNTHETIC reasoning for exclusion criterion 2. Evidence sentences: []. Label: not excluded",
            "Final label: 1",
        ]
    )
    assert format_assessment(_pair(kept, "NCT00003470").reference).endswith(
        "included\nExclusion criteria:\nFinal label: 2"
    )


def test_references_are_kept_only_when_the_derived_label_agrees(pairs: list[ClinicalPair], matching: dict) -> None:
    flipped = [dataclasses.replace(p, label=1) if p.trial_id == "NCT02777398" else p for p in pairs]
    outputs = dict(matching)
    del outputs[_pair(pairs, "NCT03727841").uid]
    outputs[_pair(pairs, "NCT00841555").uid] = {"inclusion": "```json {not valid", "exclusion": {}}
    kept, stats = attach_references(flipped, outputs)
    assert [p.trial_id for p in kept] == ["NCT00430911", "NCT04211974", "NCT00003470"]
    assert (stats.kept, stats.no_output, stats.unusable_output, stats.label_disagreement, stats.dropped) == (
        3,
        1,
        1,
        1,
        3,
    )
    assert stats.as_dict()["labels"] == {
        "pairs": {"0": 1, "1": 3, "2": 2},
        "with_output": {"0": 0, "1": 3, "2": 2},
        "kept": {"0": 0, "1": 1, "2": 2},
    }


def test_parse_matching_output_skips_malformed_entries(pairs: list[ClinicalPair]) -> None:
    trial = _pair(pairs, "NCT00430911").trial
    output = {
        "inclusion": {
            "1": ["Meets  the\nage criterion", [3, "4", True, "x"], " Included "],
            "0": ["Histology confirmed.", [0], "included"],
            "6": ["no such criterion", [], "not included"],
            "01": ["not a criterion number", [], "not included"],
            "2": ["two elements only", []],
        },
        "exclusion": {"0": ["Prior radiotherapy.", [5], "excluded"]},
    }
    assessment = parse_matching_output(output, trial)
    assert assessment == ReferenceAssessment(
        inclusion=(
            CriterionJudgment(0, "Histology confirmed.", (0,), "included"),
            CriterionJudgment(1, "Meets  the\nage criterion", (3, 4), "included"),
        ),
        exclusion=(CriterionJudgment(0, "Prior radiotherapy.", (5,), "excluded"),),
    )
    assert assessment.label == 1
    assert (
        format_judgment(assessment.inclusion[1])
        == "1. Meets the age criterion. Evidence sentences: [3, 4]. Label: included"
    )
    assert (
        format_judgment(CriterionJudgment(2, "", (), "not applicable"))
        == "2. Evidence sentences: []. Label: not applicable"
    )
    assert parse_matching_output({"inclusion": {"9": ["x", [], "included"]}, "exclusion": {}}, trial) is None
    assert parse_matching_output({"inclusion": {}}, trial) is None
    assert parse_matching_output("not a dict", trial) is None
    assert parse_matching_output({"inclusion": {}, "exclusion": {}}, trial) is None
    assert parse_matching_output({"inclusion": output["inclusion"], "exclusion": {}}, trial) is None
    no_exclusion = parse_matching_output(
        {"inclusion": {"0": ["Histology confirmed.", [0], "included"]}, "exclusion": {}},
        _pair(pairs, "NCT00003470").trial,
    )
    assert no_exclusion == ReferenceAssessment(
        inclusion=(CriterionJudgment(0, "Histology confirmed.", (0,), "included"),), exclusion=()
    )
    empty = parse_matching_output(
        {"inclusion": {}, "exclusion": {}}, {"inclusion_criteria": "", "exclusion_criteria": ""}
    )
    assert empty is not None and empty.label == 2
    superscript = {
        "inclusion": {"0": ["Histology confirmed.", [0, "²", " 3 "], "included"], "²": ["x", [], "not included"]},
        "exclusion": {"0": ["Prior radiotherapy.", [5], "excluded"]},
    }
    assessment = parse_matching_output(superscript, trial)
    assert assessment is not None
    assert assessment.inclusion == (CriterionJudgment(0, "Histology confirmed.", (0, 3), "included"),)


def test_derived_trial_label_rule() -> None:
    assert derive_trial_label(["included", "not included"], ["excluded"]) == 0
    assert derive_trial_label(["included", "not enough information"], ["not excluded", "excluded"]) == 1
    assert derive_trial_label(["included", "not applicable"], ["not excluded", "not enough information"]) == 2
    assert derive_trial_label([], []) == 2


@pytest.mark.parametrize(
    ("response", "label"),
    [
        ("Inclusion criteria:\n0. ok. Label: included\nExclusion criteria:\nFinal label: 2", 2),
        ("Final label: 0 ... on reflection,\nfinal LABEL:1", 1),
        ("**Final label:** 2", 2),
        ("**Final label**: 0.", 0),
        ("Final label : \n 1", 1),
        ("Final label: 3", None),
        ("Final label: 20", None),
        ("The final answer is 2", None),
        ("", None),
    ],
)
def test_parse_final_label(response: str, label: int | None) -> None:
    assert parse_final_label(response) == label


def test_clinical_task_verifies_final_labels() -> None:
    task = ClinicalTask(PLAIN)
    assert isinstance(task, Task) and isinstance(task, ResponseDecoder) and task.name == "clinical_trials"
    examples = [Example(uid=f"u{i}", prompt="p", reference="", target=gold) for i, gold in enumerate((2, 1, 0))]
    responses = ["... Final label: 2", "Final label: 2", "no label at all"]
    assert task.verify_batch(examples, responses) == [True, False, False]
    assert task.extract_batch(examples, responses) == [2, 2, None]
    undecided = ClinicalTask()
    assert isinstance(undecided, Task) and undecided.response_format is None
    with pytest.raises(ValueError, match="response format is not known"):
        undecided.verify_batch(examples, responses)
    with pytest.raises(ValueError, match="response_format"):
        ClinicalTask("json")


@pytest.mark.parametrize(
    ("response", "response_format", "answer", "label"),
    [
        ("Inclusion criteria:\nFinal label: 1", PLAIN, "Inclusion criteria:\nFinal label: 1", 1),
        ("I would say Final label: 2 if the note confirms", THINK, None, None),
        ("Draft: Final label: 2\n</think>\n\nInclusion criteria:", THINK, "\n\nInclusion criteria:", None),
        ("a</think>Final label: 1</think>\nFinal label: 2", THINK, "\nFinal label: 2", 2),
        ("<|channel|>analysis<|message|>Final label: 2, then check", HARMONY, None, None),
        (
            "<|channel|>analysis<|message|>Final label: 2<|end|><|start|>assistant<|channel|>final<|message|>"
            "Final label: 0<|return|>",
            HARMONY,
            "Final label: 0",
            0,
        ),
        (
            "<|channel|>final<|message|>Final label: 1<|end|><|start|>assistant<|channel|>analysis<|message|>"
            "Final label: 2",
            HARMONY,
            "Final label: 1",
            1,
        ),
        ("<|channel|>final<|message|>Inclusion criteria:\n0.", HARMONY, "Inclusion criteria:\n0.", None),
        ("analysisFinal label: 0assistantfinalFinal label: 2", HARMONY, None, None),
    ],
)
def test_only_the_final_answer_segment_is_parsed(
    response: str, response_format: str, answer: str | None, label: int | None
) -> None:
    assert final_answer(response, response_format) == answer
    assert parse_final_label(response, response_format) == label
    assert is_correct(response, 2, response_format) is (label == 2)
    examples = [Example(uid="u", prompt="p", reference="", target=2)]
    assert ClinicalTask(response_format).extract_batch(examples, [response]) == [label]


@pytest.mark.parametrize(
    ("chat_template", "response_format"),
    [
        (R1_DISTILL_TEMPLATE, THINK),
        (R1_DISTILL_TEMPLATE_WITHOUT_PREFILL, THINK),
        (CLOSED_THINK_TEMPLATE, PLAIN),
        (HARMONY_TEMPLATE, HARMONY),
        (CHAT_TEMPLATE, PLAIN),
        ("{{ raise_exception('needs a system message') }}</think>", THINK),
        ("{{ raise_exception('needs a system message') }}", PLAIN),
        (None, PLAIN),
    ],
)
def test_response_format_detected_from_the_chat_template(chat_template: str | None, response_format: str) -> None:
    tokenizer = _response_tokenizer((), (), "")
    tokenizer.chat_template = chat_template
    assert detect_response_format(tokenizer) == response_format
    assert ClinicalTask.for_tokenizer(tokenizer).response_format == response_format


@pytest.mark.parametrize("model", ["r1_distill", "r1_distill_special_think_tokens", "gpt_oss"])
def test_truncated_reasoning_never_counts_as_the_final_answer(model: str, pairs: list[ClinicalPair]) -> None:
    if model == "gpt_oss":
        tokenizer, responses, response_format, generation_prompt = (
            _harmony_tokenizer(),
            HARMONY_RESPONSES,
            HARMONY,
            "<|start|>assistant",
        )
    else:
        tokenizer = _r1_distill_tokenizer(think_is_special=model == "r1_distill_special_think_tokens")
        responses, response_format, generation_prompt = R1_RESPONSES, THINK, "<｜Assistant｜><think>\n"
    example = build_example(_pair(pairs, "NCT04211974"), tokenizer)
    assert example.target == 2 and example.prompt.endswith(generation_prompt)
    ids = _generated_ids(tokenizer, responses)
    assert [parse_final_label(text) for text in tokenizer.batch_decode(ids, skip_special_tokens=True)] == [2, 2, 2]

    task = ClinicalTask()
    texts = decode_for_verification(task, tokenizer, ids)
    assert task.response_format == response_format == ClinicalTask.for_tokenizer(tokenizer).response_format
    examples = [example] * len(ids)
    assert task.verify_batch(examples, texts) == [False, False, True]
    assert task.extract_batch(examples, texts) == [None, None, 2]
    assert texts == decode_responses(tokenizer, ids, response_format)
    if response_format == THINK:
        assert [text.count("</think>") for text in texts] == [0, 1, 1]
        assert texts[2] == R1_RESPONSES[2][0]
    else:
        assert texts[:2] == [text for text, _ in HARMONY_RESPONSES[:2]]
        assert texts[2] == HARMONY_RESPONSES[2][0] + "<|return|>"


def test_metrics_from_counts() -> None:
    gold = [2, 2, 2, 1, 1, 0, 0, 0, 0, 2]
    predicted = [2, 2, 0, 2, 1, 0, 2, 0, 1, 1]
    matrix = confusion_matrix(gold, predicted)
    assert matrix == [[2, 1, 1], [0, 1, 1], [1, 1, 2]]
    counts = binary_counts(matrix, [2])
    assert counts == {"tp": 2, "fp": 2, "fn": 2, "tn": 4}
    assert binary_metrics(**counts) == {"precision": 50.0, "recall": 50.0, "f1": 50.0, "accuracy": 60.0}
    counts = binary_counts(matrix, [1, 2])
    assert counts == {"tp": 5, "fp": 2, "fn": 1, "tn": 2}
    metrics = binary_metrics(**counts)
    assert metrics["precision"] == pytest.approx(100 * 5 / 7)
    assert metrics["recall"] == pytest.approx(100 * 5 / 6)
    p, r = metrics["precision"], metrics["recall"]
    assert metrics["f1"] == pytest.approx(2 * p * r / (p + r)) == pytest.approx(100 * 10 / 13)
    assert metrics["accuracy"] == pytest.approx(70.0)
    assert binary_metrics(tp=0, fp=0, fn=0, tn=5) == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 100.0}
    with pytest.raises(ValueError):
        binary_counts(matrix, [3])
    with pytest.raises(ValueError):
        confusion_matrix([0, 1], [0])


def test_train_validation_split_is_seeded_and_disjoint(pairs: list[ClinicalPair]) -> None:
    train, validation = split_train_validation(pairs, 0.8, seed=1111)
    assert (len(train), len(validation)) == (5, 1)
    assert sorted(p.uid for p in train + validation) == sorted(p.uid for p in pairs)
    assert [p.uid for p in train] == sorted(p.uid for p in train)
    again = split_train_validation(list(reversed(pairs)), 0.8, seed=1111)
    assert again == (train, validation)
    assert len({tuple(p.uid for p in split_train_validation(pairs, 0.8, seed)[1]) for seed in range(20)}) > 1
    assert split_train_validation(pairs, 1.0, seed=0) == (sorted(pairs, key=lambda p: p.uid), [])
    with pytest.raises(ValueError):
        split_train_validation(pairs, 0.0, seed=0)


def test_a_missing_pair_moves_at_most_one_other_pair_across_the_split() -> None:
    pairs = [
        ClinicalPair("sigir" if i % 7 == 0 else "trec_2021", f"patient-{i // 20}", f"NCT{i:08d}", i % 3, "", {})
        for i in range(500)
    ]
    train, validation = split_train_validation(pairs, 0.8, seed=1111)
    ranked = sorted(pairs, key=lambda pair: hashlib.sha256(f"1111:{pair.uid}".encode()).digest())
    assert {p.uid for p in train} == {p.uid for p in ranked[:400]} and len(validation) == 100
    for missing in (train[0], train[-1], ranked[0], ranked[399], ranked[400], validation[0], validation[-1]):
        rest = [pair for pair in pairs if pair is not missing]
        new_train, new_validation = split_train_validation(rest, 0.8, seed=1111)
        assert len(new_train) == round(0.8 * 499)
        moved = {p.uid for p in new_train} ^ {p.uid for p in train if p is not missing}
        assert len(moved) <= 1, f"{len(moved)} pairs changed sides when {missing.uid} is missing"
        assert {p.uid for p in new_train + new_validation} == {p.uid for p in rest}


def test_training_and_evaluation_pairs_from_config(pairs: list[ClinicalPair]) -> None:
    cfg = ClinicalDataConfig.from_dict(
        {
            "dataset_dir": str(FIXTURES),
            "train_corpora": "trec_2021",
            "test_corpora": [],
            "matching_results": {"trec_2021": str(MATCHING)},
            "train_fraction": 0.5,
            "split_seed": 7,
        }
    )
    train, summary = load_training_pairs(cfg)
    validation, _ = load_evaluation_pairs(cfg, "validation")
    assert len(train) == len(validation) == 3
    assert all(p.reference is not None for p in train) and all(p.reference is None for p in validation)
    assert sorted(p.uid for p in train + validation) == sorted(p.uid for p in pairs)
    labels = label_counts(train)
    assert summary["references"] == {
        "pairs": 3,
        "kept": 3,
        "no_output": 0,
        "unusable_output": 0,
        "label_disagreement": 0,
        "dropped": 0,
        "labels": {"pairs": labels, "with_output": labels, "kept": labels},
    }
    assert summary["train_labels"] == labels and sum(labels.values()) == 3
    assert summary["train_pairs"] == summary["validation_pairs"] == 3
    assert summary["corpora"]["trec_2021"]["pairs"] == 6
    assert summary["corpora"]["trec_2021"]["qrels_labels"] == {"0": 2, "1": 2, "2": 2}
    assert summary["sentence_splitter"] == sentence_splitter()
    swapped = dataclasses.replace(cfg, train_corpora=[], test_corpora=["trec_2021"])
    test, test_summary = load_evaluation_pairs(swapped, "test")
    assert [p.uid for p in test] == sorted(p.uid for p in pairs) and test_summary["test_pairs"] == 6
    assert test_summary["corpora"]["trec_2021"]["qrels_labels"] == {"0": 2, "1": 2, "2": 2}
    assert test_summary["sentence_splitter"] == sentence_splitter()
    with pytest.raises(ValueError, match="matching_results"):
        load_training_pairs(dataclasses.replace(cfg, matching_results={}))


@pytest.mark.parametrize(
    "data",
    [
        {"dataset_dir": "d", "unknown_key": 1},
        {"dataset_dir": ""},
        {"dataset_dir": "d", "train_corpora": ["trec_2021"], "test_corpora": ["trec_2021"]},
        {"dataset_dir": "d", "positive_labels": [3]},
        {"dataset_dir": "d", "positive_labels": []},
        {"dataset_dir": "d", "unparsable_label": -1},
        {"dataset_dir": "d", "train_fraction": 1.5},
        {"dataset_dir": "d", "matching_results": ["a.json"]},
        {"dataset_dir": "d", "system_in_user": "yes"},
    ],
)
def test_data_config_validation(data: dict) -> None:
    with pytest.raises(ValueError):
        ClinicalDataConfig.from_dict(data)


def test_the_system_message_can_open_the_user_turn(pairs: list[ClinicalPair]) -> None:
    pair = pairs[0]
    user = user_message(pair.patient_note, pair.trial)
    assert ClinicalDataConfig.from_dict({"dataset_dir": "d"}).system_in_user is False
    assert ClinicalDataConfig.from_dict({"dataset_dir": "d", "system_in_user": True}).system_in_user is True
    merged = build_messages(pair.patient_note, pair.trial, system_in_user=True)
    assert merged == [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user}"}]
    tokenizer = _r1_distill_tokenizer(think_is_special=False)
    default = build_example(pair, tokenizer)
    assert default.prompt == f"{R1_BOS}{SYSTEM_PROMPT}<｜User｜>{user}<｜Assistant｜><think>\n"
    (in_user,) = build_examples([pair], tokenizer, system_in_user=True)
    assert in_user.prompt == f"{R1_BOS}<｜User｜>{SYSTEM_PROMPT}\n\n{user}<｜Assistant｜><think>\n"
    assert (in_user.uid, in_user.target, in_user.meta) == (default.uid, default.target, default.meta)
    plain = _tiny_tokenizer(pairs, None)
    assert build_example(pair, plain, system_in_user=True).prompt == build_example(pair, plain).prompt


def test_trial_overlap_on_fixture_qrels(tmp_path: Path) -> None:
    train_qrels = FIXTURES / "trec_2021" / "qrels" / "test.tsv"
    test_qrels = tmp_path / "test.tsv"
    test_qrels.write_text(
        "query-id\tcorpus-id\tscore\n"
        "trec-20221\tNCT04211974\t2\n"
        "trec-20221\tNCT00003470\t0\n"
        "trec-20221\tNCT02777398\t2\n"
        "trec-20222\tNCT00841555\t1\n"
        "trec-20222\tNCT99999999\t2\n"
        "trec-20222\tNCT04211974\t0\n",
        encoding="utf-8",
    )
    report = trial_overlap(load_qrels(train_qrels), load_qrels(test_qrels), [2])
    expected_counts = {
        "test_patients": 2,
        "test_patients_in_train": 0,
        "test_trials": 5,
        "overlapping_trials": 4,
        "test_pairs": 6,
        "overlapping_pairs": 5,
        "test_positive_pairs": 3,
        "overlapping_positive_pairs": 2,
    }
    assert {key: report[key] for key in expected_counts} == expected_counts
    assert report["overlapping_trials_percent"] == pytest.approx(80.0)
    assert report["overlapping_positive_rate_percent"] == pytest.approx(40.0)
    identity = report["trial_identity_baseline"]
    assert {k: identity[k] for k in ("tp", "fp", "fn", "tn")} == {"tp": 1, "fp": 2, "fn": 1, "tn": 1}
    assert (identity["precision"], identity["recall"], identity["f1"]) == pytest.approx((100 / 3, 50.0, 40.0))
    everything = report["all_positive_baseline"]
    assert {k: everything[k] for k in ("tp", "fp", "fn", "tn")} == {"tp": 2, "fp": 3, "fn": 0, "tn": 0}
    assert everything["f1"] == pytest.approx(100 * 4 / 7)
    wider = trial_overlap(load_qrels(train_qrels), load_qrels(test_qrels), [1, 2])["trial_identity_baseline"]
    assert {k: wider[k] for k in ("tp", "fp", "fn", "tn")} == {"tp": 2, "fp": 2, "fn": 1, "tn": 0}
    output = tmp_path / "report" / "overlap.json"
    cli = overlap_main(["--train_qrels", str(train_qrels), "--test_qrels", str(test_qrels), "--output", str(output)])
    assert cli == report == json.loads(output.read_text(encoding="utf-8"))


MAIN_CONFIGS = {
    "trec2022_deepseek_r1_32b.yaml": (
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        ["sigir", "trec_2021"],
        ["trec_2022"],
    ),
    "trec2022_gptoss_20b.yaml": ("openai/gpt-oss-20b", ["sigir", "trec_2021"], ["trec_2022"]),
    "trec2022_deepseek_r1_7b.yaml": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", ["sigir", "trec_2021"], ["trec_2022"]),
    "trec2021_gptoss_20b_out_of_year.yaml": ("openai/gpt-oss-20b", ["sigir", "trec_2022"], ["trec_2021"]),
}
OVERLAYS = {
    "arix_wo_rb": {"method": "arix_wo_rb", "use_bank": False, "use_contrastive": True, "use_partial_reward": True},
    "grpo": {"method": "grpo", "use_bank": False, "use_contrastive": False, "use_partial_reward": False},
    "random_retrieval": {"method": "arix", "use_bank": True, "retrieval": "random"},
    "fifo_update": {"method": "arix", "use_bank": True, "bank_update": "fifo"},
    "random_update": {"method": "arix", "use_bank": True, "bank_update": "random"},
    "no_partial_reward": {"method": "arix_wo_rb", "use_partial_reward": False, "use_contrastive": True},
    "no_contrastive": {"method": "arix_wo_rb", "use_contrastive": False, "use_partial_reward": True},
    "negative_loss": {"method": "arix_wo_rb", "use_bank": False, "use_negative": True, "alpha_neg": 5e-4},
}


@pytest.mark.parametrize("name", sorted(MAIN_CONFIGS))
def test_main_configs_hold_the_manuscript_hyperparameters(name: str) -> None:
    assert sorted(p.name for p in CONFIGS.glob("*.yaml")) == sorted(MAIN_CONFIGS)
    model, train_corpora, test_corpora = MAIN_CONFIGS[name]
    cfg = load_run_config([str(CONFIGS / name)], [])
    data = ClinicalDataConfig.from_dict(cfg.data)
    t, a = cfg.training, cfg.arix
    assert (cfg.method, cfg.model.name_or_path) == ("arix", model)
    assert (t.epochs, t.learning_rate, t.batch_size, t.group_size) == (10, 1e-6, 32, 64)
    assert (a.use_bank, a.use_contrastive, a.use_partial_reward, a.use_negative) == (True, True, True, False)
    assert (a.alpha_pos, a.bank_size, a.partial_scale, a.partial_bias, a.proj_dim) == (1e-3, 5, 0.1, 0.2, 64)
    assert (a.retrieval, a.bank_update, a.retrieval_start_epoch, a.pooling) == (
        "similarity",
        "redundancy",
        2,
        "softmax",
    )
    assert cfg.generation.max_new_tokens == 2048
    assert (data.train_corpora, data.test_corpora) == (train_corpora, test_corpora)
    assert sorted(data.matching_results) == sorted(train_corpora)
    assert (data.train_fraction, data.positive_labels, data.unparsable_label) == (0.8, [2], 0)
    assert data.system_in_user is False
    assert cfg.output.output_dir == f"outputs/clinical/{Path(name).stem}"


@pytest.mark.parametrize("name", sorted(OVERLAYS))
def test_ablation_overlays(name: str) -> None:
    assert sorted(p.stem for p in (CONFIGS / "ablations").glob("*.yaml")) == sorted(OVERLAYS)
    base = str(CONFIGS / "trec2022_gptoss_20b.yaml")
    cfg = load_run_config([base, str(CONFIGS / "ablations" / f"{name}.yaml")], [])
    expected = dict(OVERLAYS[name])
    assert cfg.method == expected.pop("method")
    assert {key: getattr(cfg.arix, key) for key in expected} == expected
    assert (cfg.training.learning_rate, cfg.training.group_size, cfg.arix.alpha_pos) == (1e-6, 64, 1e-3)
    assert cfg.output.output_dir == f"outputs/clinical/trec2022_gptoss_20b-{name}"
    explicit = load_run_config([base, str(CONFIGS / "ablations" / f"{name}.yaml")], ["output.output_dir=elsewhere"])
    assert explicit.output.output_dir == "elsewhere"


def test_evaluate_end_to_end_with_a_tiny_model(tmp_path: Path, pairs: list[ClinicalPair]) -> None:
    tokenizer = _tiny_tokenizer(pairs, CHAT_TEMPLATE)
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=8192,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    model_dir = tmp_path / "model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {"name_or_path": str(model_dir), "torch_dtype": "float32"},
                "data": {"dataset_dir": str(FIXTURES), "train_corpora": [], "test_corpora": ["trec_2021"]},
                "generation": {"max_new_tokens": 4},
                "output": {"output_dir": str(tmp_path / "run")},
            }
        ),
        encoding="utf-8",
    )
    common = ["--model", str(model_dir), "--config", str(config), "--batch_size", "4"]
    predictions = tmp_path / "eval" / "test.jsonl"
    summary = evaluate.main([*common, "--output_file", str(predictions)])
    records = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    assert [r["uid"] for r in records] == sorted(p.uid for p in pairs) and [r["index"] for r in records] == list(
        range(6)
    )
    assert all(len(r["responses"]) == len(r["parsed_labels"]) == 1 and r["num_tokens"][0] <= 4 for r in records)
    assert summary is not None and (summary["num_pairs"], summary["num_evaluated"]) == (6, 6)
    assert sum(map(sum, summary["confusion_matrix"])) == 6 and [sum(row) for row in summary["confusion_matrix"]] == [
        2,
        2,
        2,
    ]
    assert summary["tp"] + summary["fp"] + summary["fn"] + summary["tn"] == 6
    unparsable = sum(r["parsed_labels"][0] is None for r in records)
    assert summary["num_unparsable"] == unparsable and summary["positive_labels"] == [2]
    assert json.loads(predictions.with_suffix(".metrics.json").read_text(encoding="utf-8")) == summary

    evaluation_files = predictions.read_text(encoding="utf-8"), predictions.with_suffix(".metrics.json").read_bytes()
    rescored = evaluate.main(
        [*common, "--rescore", str(predictions), "--positive_labels", "1", "2", "--unparsable_label", "2"]
    )
    assert rescored is not None and rescored["positive_labels"] == [1, 2] and rescored["model"] == str(model_dir)
    rescored_file = tmp_path / "eval" / "test.rescore_pos1-2.metrics.json"
    assert json.loads(rescored_file.read_text(encoding="utf-8")) == rescored
    assert (predictions.read_text(encoding="utf-8"), predictions.with_suffix(".metrics.json").read_bytes()) == (
        evaluation_files
    )
    parsed = [r["parsed_labels"][0] for r in records]
    rescored_labels = [2 if label is None else label for label in parsed]
    assert rescored["confusion_matrix"] == confusion_matrix([r["label"] for r in records], rescored_labels)
    assert rescored["num_unparsable"] == unparsable and rescored["unparsable_label"] == 2

    diverse = tmp_path / "eval" / "diversity.jsonl"
    summary = evaluate.main(
        [
            *common,
            "--num_samples",
            "2",
            "--temperature",
            "1.0",
            "--diversity",
            "--max_pairs",
            "2",
            "--output_file",
            str(diverse),
        ]
    )
    assert summary is not None and (summary["num_pairs"], summary["num_evaluated"]) == (2, 4)
    assert 0.0 <= summary["self_bleu"] <= 1.0 and 0.0 <= summary["mean_pairwise_cosine_distance"] <= 2.0
    assert all("cosine_distance" in json.loads(line) for line in diverse.read_text(encoding="utf-8").splitlines())


SCRIPTED_ANSWERS = {
    "NCT02777398": ("Final label: 0", True),
    "NCT03727841": ("The note does not say whether", False),
    "NCT00841555": ("Inclusion criteria:\n0. Glioblastoma confirmed.\nFinal label: 1", True),
    "NCT00430911": ("Final label: 2", True),
    "NCT04211974": ("Final label: **2**", True),
    "NCT00003470": ("Final label: 1", True),
}


def _script_clinical_generation(monkeypatch: pytest.MonkeyPatch, pairs: list[ClinicalPair], respond: Any) -> list[int]:
    calls: list[int] = []

    def generate(_model: Any, tokenizer: Any, prompts: list[list[int]], *_settings: Any) -> list[list[int]]:
        calls.append(len(prompts))
        rollouts: dict[tuple[int, ...], int] = {}
        responses = []
        for prompt in prompts:
            text = tokenizer.decode(prompt)
            pair = next(pair for pair in pairs if pair.trial["brief_title"] in text)
            number = rollouts[tuple(prompt)] = rollouts.get(tuple(prompt), -1) + 1
            responses += _generated_ids(tokenizer, [respond(pair, number)])
        return responses

    monkeypatch.setattr(arix.evaluation, "generate_batch", generate)
    return calls


def _evaluation_config(tmp_path: Path, model_dir: Path, dataset_dir: Path = FIXTURES, **sections: Any) -> Path:
    path = tmp_path / f"{model_dir.name}_config.yaml"
    run = {
        "model": {"name_or_path": str(model_dir), "torch_dtype": "float32"},
        "data": {"dataset_dir": str(dataset_dir), "train_corpora": [], "test_corpora": ["trec_2021"]},
        "generation": {"max_new_tokens": 16},
        "output": {"output_dir": str(tmp_path / "run")},
        **sections,
    }
    path.write_text(yaml.safe_dump(run), encoding="utf-8")
    return path


def test_evaluate_scores_scripted_labels_exactly(
    tiny_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pairs: list[ClinicalPair],
) -> None:
    calls = _script_clinical_generation(monkeypatch, pairs, lambda pair, _rollout: SCRIPTED_ANSWERS[pair.trial_id])
    config = _evaluation_config(tmp_path, tiny_model_dir, evaluation={"batch_size": 4})
    common = ["--model", str(tiny_model_dir), "--config", str(config)]
    predictions = tmp_path / "eval" / "test.jsonl"
    summary = evaluate.main([*common, "--output_file", str(predictions)])
    assert summary is not None and calls == [4, 2]
    assert "6 pairs evaluated of 6 qrels judgments (trec_2021)" in capsys.readouterr().out
    assert summary["confusion_matrix"] == [[2, 0, 0], [0, 1, 1], [0, 1, 1]]
    assert {key: summary[key] for key in ("tp", "fp", "fn", "tn")} == {"tp": 1, "fp": 1, "fn": 1, "tn": 3}
    assert (summary["precision"], summary["recall"], summary["f1"]) == (50.0, 50.0, 50.0)
    assert summary["accuracy"] == pytest.approx(400 / 6) and summary["accuracy_3class"] == pytest.approx(400 / 6)
    assert (summary["num_pairs"], summary["num_unparsable"], summary["num_truncated"]) == (6, 1, 1)
    assert (summary["batch_size"], summary["response_format"], summary["positive_labels"]) == (4, PLAIN, [2])

    records = {
        record["trial_id"]: record for record in map(json.loads, predictions.read_text(encoding="utf-8").splitlines())
    }
    assert [records[trial]["parsed_labels"] for trial in SCRIPTED_ANSWERS] == [[0], [None], [1], [2], [2], [1]]
    verdicts = [records[trial]["correct"][0] for trial in SCRIPTED_ANSWERS]
    assert verdicts == [True, False, True, False, True, False]
    scored = [records[trial]["scored_correct"][0] for trial in SCRIPTED_ANSWERS]
    assert scored == [True, True, True, False, True, False]
    assert 100 * sum(scored) / len(scored) == pytest.approx(summary["accuracy_3class"])
    assert [records[trial]["finished"][0] for trial in SCRIPTED_ANSWERS] == [True, False, True, True, True, True]

    metrics = predictions.with_suffix(".metrics.json")
    written = metrics.read_bytes(), predictions.read_bytes()
    calls.clear()
    rescored = evaluate.main([*common, "--rescore", str(predictions), "--positive_labels", "1", "2"])
    assert rescored is not None and not calls
    assert json.loads((tmp_path / "eval" / "test.rescore_pos1-2.metrics.json").read_text(encoding="utf-8")) == rescored
    assert (metrics.read_bytes(), predictions.read_bytes()) == written
    assert {key: rescored[key] for key in ("tp", "fp", "fn", "tn")} == {"tp": 4, "fp": 0, "fn": 0, "tn": 2}
    assert (rescored["precision"], rescored["recall"], rescored["f1"]) == (100.0, 100.0, 100.0)
    assert rescored["confusion_matrix"] == summary["confusion_matrix"] and rescored["rescored_from"] == str(predictions)
    assert (rescored["model"], rescored["batch_size"], rescored["data"]) == (str(tiny_model_dir), 4, summary["data"])

    elsewhere = tmp_path / "rescored.json"
    unparsable_as_2 = ["--unparsable_label", "2", "--output_file", str(elsewhere)]
    rescored = evaluate.main([*common, "--rescore", str(predictions), "--positive_labels", "1", "2", *unparsable_as_2])
    assert rescored is not None and json.loads(elsewhere.read_text(encoding="utf-8")) == rescored
    assert rescored["confusion_matrix"] == [[1, 0, 1], [0, 1, 1], [0, 1, 1]]
    assert {key: rescored[key] for key in ("tp", "fp", "fn", "tn")} == {"tp": 4, "fp": 1, "fn": 0, "tn": 1}
    assert (rescored["precision"], rescored["recall"], rescored["f1"]) == (80.0, 100.0, pytest.approx(800 / 9))
    with pytest.raises(ValueError, match="overwrite"):
        evaluate.main([*common, "--rescore", str(predictions), "--output_file", str(metrics)])
    monkeypatch.setenv("RANK", "1")
    assert evaluate.main([*common, "--rescore", str(predictions), "--positive_labels", "0"]) is None
    assert not (tmp_path / "eval" / "test.rescore_pos0.metrics.json").exists()
    assert (metrics.read_bytes(), predictions.read_bytes()) == written


def test_the_test_split_scores_every_judgment(
    tiny_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pairs: list[ClinicalPair],
) -> None:
    dataset = _copy_corpus(tmp_path, "queries.jsonl", "corpus.jsonl")
    qrels = dataset / "trec_2021" / "qrels" / "test.tsv"
    qrels.write_text(qrels.read_text(encoding="utf-8") + "trec-20211\tNCT99999999\t2\n", encoding="utf-8")
    config = _evaluation_config(tmp_path, tiny_model_dir, dataset_dir=dataset)
    common = ["--model", str(tiny_model_dir), "--config", str(config), "--output_file", str(tmp_path / "test.jsonl")]
    with pytest.raises(ValueError, match="1 judgments without trial record.*--allow_missing_pairs"):
        evaluate.main(common)
    _script_clinical_generation(monkeypatch, pairs, lambda pair, _rollout: (f"Final label: {pair.label}", True))
    summary = evaluate.main([*common, "--allow_missing_pairs"])
    assert summary is not None and (summary["num_pairs"], summary["accuracy_3class"]) == (6, 100.0)
    assert summary["data"]["corpora"]["trec_2021"]["missing_trial"] == 1
    assert "6 pairs evaluated of 7 qrels judgments (trec_2021)" in capsys.readouterr().out


def test_gpt_oss_responses_are_read_from_their_final_channel(
    tiny_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    pairs: list[ClinicalPair],
) -> None:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tiny_model_dir)
    tokenizer.add_tokens([AddedToken(token, special=True, normalized=False) for token in HARMONY_TOKENS])
    tokenizer.eos_token, tokenizer.chat_template = "<|return|>", HARMONY_TEMPLATE
    model_dir = tmp_path / "gpt_oss"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=len(tokenizer),
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=16384,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        )
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    config = _evaluation_config(tmp_path, model_dir)
    _script_clinical_generation(monkeypatch, pairs, lambda _pair, rollout: HARMONY_RESPONSES[rollout])
    predictions = tmp_path / "eval" / "harmony.jsonl"
    sampling = ["--num_samples", "3", "--temperature", "1.0", "--diversity", "--max_pairs", "1"]
    sampling += ["--embed_batch_size", "1"]
    common = ["--model", str(model_dir), "--config", str(config)]
    summary = evaluate.main([*common, *sampling, "--output_file", str(predictions)])
    [record] = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]

    ids = _generated_ids(tokenizer, HARMONY_RESPONSES)
    assert summary is not None and summary["response_format"] == record["response_format"] == HARMONY
    assert record["responses"] == tokenizer.batch_decode(ids, skip_special_tokens=False)
    assert record["parsed_labels"] == [None, None, 2] and summary["num_unparsable"] == 2
    assert record["correct"] == [False, False, record["label"] == 2]
    without_markers = tokenizer.batch_decode(ids, skip_special_tokens=True)
    assert record["self_bleu"] == pytest.approx(self_bleu(without_markers))
    assert record["self_bleu"] != pytest.approx(self_bleu(record["responses"]))
    assert (summary["self_bleu_n"], summary["self_bleu_tokenizer"], summary["embed_batch_size"]) == (4, "regex", 1)

    rescored = evaluate.main([*common, "--rescore", str(predictions)])
    assert rescored is not None and rescored.pop("rescored_from") == str(predictions) and rescored == summary

    stripped = tmp_path / "eval" / "unrecorded.jsonl"
    unrecorded = {key: value for key, value in record.items() if key != "response_format"}
    stripped.write_text(json.dumps(unrecorded) + "\n", encoding="utf-8")
    with caplog.at_level("WARNING"):
        rescored = evaluate.main([*common, "--rescore", str(stripped)])
    assert rescored is not None and (rescored["response_format"], rescored["num_unparsable"]) == (PLAIN, 0)
    assert "1 of 1 records have no response_format" in caplog.text


def test_the_summary_line_compares_test_pairs_with_the_qrels_judgments_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from clinical_trials.evaluate import print_summary

    metrics = {"num_evaluated": 1, "num_unparsable": 0, "positive_labels": [2], "precision": 100.0}
    metrics |= {"recall": 100.0, "f1": 100.0, "accuracy": 100.0}
    corpora = {"corpora": {"trec_2021": {"qrels": 6}}}
    print_summary({**metrics, "split": "test", "num_pairs": 6, "data": corpora})
    assert "6 pairs evaluated of 6 qrels judgments (trec_2021)," in capsys.readouterr().out
    print_summary({**metrics, "split": "validation", "num_pairs": 1, "data": corpora})
    assert "1 pair evaluated (validation split)," in capsys.readouterr().out


def test_the_default_output_name_holds_the_split_corpora_subset_and_decoding() -> None:
    assert evaluate.output_name("test", ["trec_2022"], None, 1) == "test_trec_2022"
    assert evaluate.output_name("validation", ["sigir", "trec_2021"], None, 1) == "validation_sigir_trec_2021"
    assert evaluate.output_name("test", ["trec_2022"], 1000, 16, 1.0) == "test_trec_2022_pairs1000_samples16"
    assert evaluate.output_name("test", ["trec_2022"], None, 1, 0.6, 0.95) == "test_trec_2022_t0.6_p0.95"
    assert evaluate.output_name("test", ["trec_2022"], None, 1, 1.0) == "test_trec_2022_t1"


def test_a_subset_or_a_sampled_evaluation_keeps_the_full_greedy_outputs(
    tiny_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pairs: list[ClinicalPair]
) -> None:
    checkpoint = tmp_path / "final"
    shutil.copytree(tiny_model_dir, checkpoint)
    shutil.copy(_evaluation_config(tmp_path, checkpoint), checkpoint / "config.yaml")
    _script_clinical_generation(monkeypatch, pairs, lambda pair, _rollout: SCRIPTED_ANSWERS[pair.trial_id])
    full_run = evaluate.main(["--model", str(checkpoint)])
    full = checkpoint / "eval" / "test_trec_2021.jsonl"
    written = full.read_bytes(), full.with_suffix(".metrics.json").read_bytes()
    assert full_run is not None and (full_run["num_pairs"], full_run["max_pairs"]) == (6, None)

    subset = evaluate.main(["--model", str(checkpoint), "--max_pairs", "2"])
    sampled = evaluate.main(["--model", str(checkpoint), "--temperature", "0.6", "--top_p", "0.95"])
    diversity = ["--num_samples", "2", "--temperature", "1.0", "--diversity", "--max_pairs", "3"]
    sampled_subset = evaluate.main(["--model", str(checkpoint), *diversity])
    assert (full.read_bytes(), full.with_suffix(".metrics.json").read_bytes()) == written
    assert subset is not None and (subset["num_pairs"], subset["max_pairs"]) == (2, 2)
    assert sampled is not None and (sampled["num_pairs"], sampled["temperature"], sampled["top_p"]) == (6, 0.6, 0.95)
    assert sampled_subset is not None and (sampled_subset["num_pairs"], sampled_subset["max_pairs"]) == (3, 3)
    every_pair = evaluate.main(["--model", str(checkpoint), "--max_pairs", "6"])
    assert every_pair is not None and every_pair["max_pairs"] is None
    assert (full.read_bytes(), full.with_suffix(".metrics.json").read_bytes()) == written
    stems = ["test_trec_2021", "test_trec_2021_pairs2", "test_trec_2021_pairs3_samples2", "test_trec_2021_t0.6_p0.95"]
    assert sorted(path.name for path in full.parent.iterdir()) == sorted(
        f"{stem}{suffix}" for stem in stems for suffix in (".jsonl", ".metrics.json")
    )
