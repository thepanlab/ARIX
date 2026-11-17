from __future__ import annotations

import contextlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from math_reasoning.data import MathProblem, load_aime, load_math, load_problems, load_reft, problem_uid
from math_reasoning.prompts import build_prompt, nl_reference, program_header, python_reference
from math_reasoning.task import MathTask, build_examples, make_example, task_from_config
from math_reasoning.verifier import (
    WORKER_SCRIPT,
    ProgramExecutor,
    extract_math_answer,
    extract_nl_answer,
    is_equiv,
    last_boxed_content,
    math_equal,
    process_code,
)

CODE_DIR = Path(__file__).parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "math"
CONFIGS = CODE_DIR / "math_reasoning" / "configs"
PYTHON_FILE = FIXTURES / "gsm8k_python_sdp_sample.json"
NL_FILE = FIXTURES / "gsm8k_nl_sample.json"
MATH_FILE = FIXTURES / "math_sample.json"


def _records(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _examples(dataset: str, engine: str, path: Path):
    return build_examples({"dataset": dataset, "engine": engine, "train_file": str(path)}, split="train")


@pytest.fixture(scope="module")
def python_task():
    task = MathTask("python", "gsm8k", timeout=1.0, num_workers=4)
    yield task
    task.close()


def test_prompt_and_reference_formatting():
    nl_record, program_record = _records(NL_FILE)[0], _records(PYTHON_FILE)[0]
    question = nl_record["question"]
    assert build_prompt(question) == f"Question:\n{question}\nAnswer reasoning:\n"
    header = f'def solution():\n    """{question}"""\n'
    assert program_header(question) == header
    assert build_prompt(question, "python") == f"Question:\n{question}\nAnswer reasoning:\n{header}"

    nl = make_example(load_reft(NL_FILE)[0], "gsm8k", "nl", with_reference=True)
    assert nl.prompt == build_prompt(question)
    assert nl.reference == nl_record["answer_cot"].strip() + "\nTherefore, the answer is: 72"
    assert nl.target == "72" and nl.uid == "gsm8k_0"
    assert nl.meta["add_special_tokens"] is True

    program = make_example(load_reft(PYTHON_FILE)[0], "gsm8k", "python", with_reference=True)
    assert program.prompt.endswith(header)
    assert program.reference.startswith("    clips_april = 48\n")
    assert program_header(question) + program.reference == program_record["answer_cot"]

    math_record = _records(MATH_FILE)[0]
    math = _examples("math", "nl", MATH_FILE)[0]
    assert math.prompt == f"Question:\n{math_record['problem']}\nAnswer reasoning:\n"
    assert math.reference == math_record["solution"].strip()
    assert math.target == math_record["answer"]


def test_python_reference_whose_docstring_differs_from_the_question():
    program = 'def solution():\n    """How many\n    apples?"""\n    result = 3\n    return result'
    assert python_reference(program, "How many apples?") == "    result = 3\n    return result"
    with pytest.raises(ValueError):
        python_reference("result = 3", "How many apples?")
    with pytest.raises(ValueError):
        nl_reference(program, "3")


def test_evaluation_examples_need_no_reference():
    problem = MathProblem(uid="gsm8k_0", question="What is 1 + 1?", answer="2")
    example = make_example(problem, "gsm8k", "python", with_reference=False)
    assert example.reference == "" and example.prompt.endswith(program_header("What is 1 + 1?"))
    with pytest.raises(ValueError):
        make_example(problem, "gsm8k", "python", with_reference=True)
    with pytest.raises(ValueError):
        make_example(MathProblem("x", "q", "two"), "gsm8k", "nl", with_reference=False)


def test_every_fixture_program_executes_to_its_answer_value(python_task):
    records = _records(PYTHON_FILE)
    results = python_task.executor.run([record["answer_cot"] for record in records])
    for record, result in zip(records, results, strict=True):
        assert result.error is None, (record["item_id"], result.error)
        assert result.value == pytest.approx(float(record["answer_value"]), abs=1e-2)


def test_python_references_verify_through_the_prompt_header(python_task):
    examples = _examples("gsm8k", "python", PYTHON_FILE)
    values, correct = python_task.score_batch(examples, [example.reference for example in examples])
    assert all(correct)
    assert values == pytest.approx([float(example.target) for example in examples], abs=1e-2)


def test_python_wrong_and_failing_programs_are_rejected(python_task):
    example = _examples("gsm8k", "python", PYTHON_FILE)[0]
    responses = [
        "    return 73",
        "    return clips",
        "    result = 72",
        "    return 'seventy-two'",
        "    import os\n    os._exit(1)",
        "    result = 72\n\n    return result",
        example.reference,
    ]
    assert python_task.verify_batch([example] * len(responses), responses) == [False] * 6 + [True]


def test_infinite_loop_times_out_and_is_rejected(python_task):
    example = _examples("gsm8k", "python", PYTHON_FILE)[0]
    start = time.monotonic()
    verdicts = python_task.verify_batch(
        [example] * 2, ["    while True:\n        pass\n    return 72", example.reference]
    )
    assert verdicts == [False, True]
    assert time.monotonic() - start < 15
    result = python_task.executor.run([program_header("q") + "    while True:\n        pass"])[0]
    assert result.value is None and "timed out" in result.error
    assert python_task.verify_batch([example], [example.reference]) == [True]


def test_programs_run_outside_the_trainer_with_restricted_globals(python_task):
    run = python_task.executor.run
    pid = run(["def solution():\n    import os\n    return os.getpid()"])[0].value
    assert pid is not None and int(pid) != os.getpid()
    noisy = run(["def solution():\n    print('noise')\n    return 1", "def solution():\n    return input()"])
    assert noisy[0].value == 1.0 and noisy[1].value is None
    assert run(["def solution():\n    return math.sqrt(16)"])[0].value == 4.0
    if python_task.executor.sympy_available:
        assert run(["def solution():\n    return sympy.Rational(1, 4)"])[0].value == 0.25


def test_program_state_does_not_leak_into_later_calls():
    executor = ProgramExecutor(num_workers=1, timeout=5.0)
    try:
        patch, read = "def solution():\n    math.pi = 3.14\n    return 1", "def solution():\n    return math.pi"
        assert [result.value for result in executor.run([patch, read])] == [1.0, 3.14]
        assert executor.run([read])[0].value == math.pi
    finally:
        executor.close()


def _running_since(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    fields = stat.rsplit(")", 1)[1].split()
    return None if fields[0] in ("Z", "X") else fields[19]


def test_worker_exits_at_once_when_its_parent_is_already_gone():
    not_the_parent = str(os.getpid() + 1)
    command = [sys.executable, "-I", str(WORKER_SCRIPT), not_the_parent]
    worker = subprocess.run(command, input="", capture_output=True, text=True, timeout=60, check=False)
    assert worker.stdout == ""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the parent-death signal (prctl) is Linux-only")
def test_worker_dies_with_its_parent_while_a_program_blocks_in_c_code(tmp_path):
    pid_file, scratch = tmp_path / "worker.pid", tmp_path / "worker.tmp"
    program = (
        "def solution():\n"
        "    import os\n"
        f"    with open({str(scratch)!r}, 'w') as handle:\n"
        "        handle.write(str(os.getpid()))\n"
        f"    os.replace({str(scratch)!r}, {str(pid_file)!r})\n"
        "    return 10 ** (10 ** 9) % 7"
    )
    starter = (
        "import sys\n"
        f"sys.path.insert(0, {str(CODE_DIR)!r})\n"
        "from math_reasoning.verifier import ProgramExecutor\n"
        "ProgramExecutor(num_workers=1, timeout=3600).run(sys.argv[1:])"
    )
    parent = subprocess.Popen([sys.executable, "-c", starter, program])
    worker, started = 0, None
    try:
        deadline = time.monotonic() + 60
        while not pid_file.exists() and parent.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid_file.exists(), "the program did not start"
        worker = int(pid_file.read_text(encoding="utf-8"))
        started = _running_since(worker)
        assert started is not None
        time.sleep(0.2)
        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=60)
        deadline = time.monotonic() + 10
        while _running_since(worker) == started and time.monotonic() < deadline:
            time.sleep(0.05)
        assert _running_since(worker) != started, "the worker outlived its parent"
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if started is not None and _running_since(worker) == started:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(worker, signal.SIGKILL)


def test_process_code_follows_reft():
    code = "def solution():\n    x = 1\n    sys.exit(0)\n    return x\n\n    print('after a blank line')"
    expected = "def solution():\n    x = 1\n    print(0)\n    return x"
    assert process_code(code, import_sympy=True) == "import math\nimport sympy\n" + expected
    assert process_code(code, import_sympy=False) == "import math\n" + expected


def test_executor_rejects_bad_settings():
    with pytest.raises(ValueError):
        ProgramExecutor(num_workers=0)
    with pytest.raises(ValueError):
        ProgramExecutor(timeout=0)


def test_nl_fixture_references_parse_to_their_answer_value():
    examples = _examples("gsm8k", "nl", NL_FILE)
    task = MathTask("nl", "gsm8k")
    values, correct = task.score_batch(examples, [example.reference for example in examples])
    assert all(correct)
    assert values == [float(record["answer_value"]) for record in _records(NL_FILE)]


def test_nl_extraction_and_tolerance():
    trigger = "\nTherefore, the answer is: "
    assert extract_nl_answer(f"so{trigger}1,234") == 1234.0
    assert extract_nl_answer(f"a{trigger}7{trigger}8") == 8.0
    assert extract_nl_answer(f"so{trigger}1e3") == 1000.0
    for decorated in ("$18", "18 dollars.", "**18**", "\\boxed{18}", "18.\n\nQuestion: If she had 20 apples"):
        assert extract_nl_answer(f"so{trigger}{decorated}") == 18.0, decorated
    assert extract_nl_answer(f"so{trigger}3/4") is None
    assert extract_nl_answer(f"so{trigger}twelve") is None
    assert extract_nl_answer("3 apples and 12.5 pears") == 12.5
    assert extract_nl_answer("x = 3-2") == 2.0
    assert extract_nl_answer("abc um5 xyz") is None
    assert extract_nl_answer("answer: 3/4") is None
    assert extract_nl_answer("no number") is None
    example = _examples("gsm8k", "nl", NL_FILE)[0]
    task = MathTask("nl", "gsm8k")
    responses = [f"x{trigger}73", f"x{trigger}72.005", f"x{trigger}72.02", "The answer is <72>.", f"x{trigger}$72."]
    assert task.verify_batch([example] * 5, responses) == [False, True, False, True, True]


def test_math_fixture_solutions_verify_against_their_answer():
    examples = _examples("math", "nl", MATH_FILE)
    task = MathTask("nl", "math")
    answers, correct = task.score_batch(examples, [example.reference for example in examples])
    assert all(correct)
    assert [example.target for example in examples] == [record["answer"] for record in _records(MATH_FILE)]
    assert answers[1] == "180\\text{ miles}"
    assert task.verify_batch(examples[:1], ["so $\\boxed{17}$"]) == [False]


def test_boxed_extraction():
    assert last_boxed_content("so $\\boxed{\\frac{1}{2}}$ and") == "\\frac{1}{2}"
    assert last_boxed_content("\\boxed{1} then \\boxed{2}") == "2"
    assert last_boxed_content("\\fbox{5}") == "5"
    assert last_boxed_content("answer \\boxed 7$.") == "7"
    assert last_boxed_content("\\boxed{unbalanced") is None
    assert last_boxed_content("no box") is None
    assert extract_math_answer("work\nTherefore, the answer is: $\\frac{1}{4}$.") == "\\frac{1}{4}"
    assert extract_math_answer("no final answer") is None


@pytest.mark.parametrize(
    ("prediction", "gold"),
    [
        ("\\dfrac{1}{2}", "\\frac12"),
        ("0.5", "\\frac{1}{2}"),
        ("x = 5", "5"),
        ("10\\%", "10"),
        ("\\$5", "5"),
        ("54^\\circ", "54"),
        ("180\\text{ miles}", "180"),
        ("\\left(1,2\\right)", "(1,2)"),
        ("1/3", "\\frac{1}{3}"),
        ("\\sqrt3", "\\sqrt{3}"),
        ("070", "70"),
        ("1,000", "1000"),
        ("0.25", "\\frac{1}{4}"),
        ("-\\frac{1}{2}", "-0.5"),
    ],
)
def test_equivalent_math_answers(prediction, gold):
    assert math_equal(prediction, gold)


@pytest.mark.parametrize(
    ("prediction", "gold"),
    [("17", "16"), ("\\frac{1}{3}", "0.33"), ("1,2", "12"), ("(1,2)", "(2,1)"), (None, "1"), ("1", None)],
)
def test_different_math_answers_are_rejected(prediction, gold):
    assert not math_equal(prediction, gold)


def test_is_equiv_falls_back_to_raw_strings_like_hendrycks():
    two_units = "1\\text{ m}\\text{ s}"
    assert is_equiv(two_units, two_units)
    assert not is_equiv(two_units, "1")


def test_reft_loader():
    problems = load_reft(PYTHON_FILE)
    assert [p.uid for p in problems] == [record["item_id"] for record in _records(PYTHON_FILE)]
    assert problems[0].answer == "72" and problems[0].solution.startswith("def solution():")
    assert load_problems(NL_FILE, "svamp")[3].answer == "42"


def test_math_loaders_and_stable_uids(tmp_path):
    records = _records(MATH_FILE)[:2]
    for number, record in enumerate(records, start=1):
        path = tmp_path / "train" / "prealgebra" / f"{number}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({key: record[key] for key in ("problem", "level", "solution")}), encoding="utf-8")
    from_directory = load_math(tmp_path / "train")
    assert [p.answer for p in from_directory] == ["16", "180\\text{ miles}"]
    assert from_directory[0].uid == problem_uid("math", records[0]["problem"])

    jsonl = tmp_path / "math.jsonl"
    jsonl.write_text("\n".join(json.dumps(record) for record in [records[0], records[1], records[0]]), encoding="utf-8")
    from_jsonl = load_math(jsonl)
    assert [p.uid for p in from_jsonl] == [from_directory[0].uid, from_directory[1].uid, from_directory[0].uid + "-1"]
    assert [p.answer for p in from_jsonl] == [record["answer"] for record in (records[0], records[1], records[0])]

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text(json.dumps({**records[0], "answer": "a sentence, not an answer"}), encoding="utf-8")
    assert load_math(malformed)[0].answer == "16"


def test_aime_loader_and_task(tmp_path):
    path = tmp_path / "aime_2025.jsonl"
    record = {"question": "Find the answer.", "answer": 70, "solution": "a worked solution with \\boxed{7}"}
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    problems = load_aime(path)
    assert problems[0].answer == "70" and problems[0].uid == problem_uid("aime", "Find the answer.")
    task = MathTask("nl", "aime")
    example = make_example(problems[0], "aime", "nl", with_reference=False)
    assert task.verify_batch([example] * 2, ["so \\boxed{070}", "so \\boxed{71}"]) == [True, False]


def test_data_config_validation():
    with pytest.raises(ValueError):
        task_from_config({"dataset": "math", "engine": "python"})
    with pytest.raises(ValueError):
        task_from_config({"dataset": "gsm8k", "trian_file": "x.json"})
    with pytest.raises(ValueError):
        build_examples({"dataset": "gsm8k", "engine": "nl"}, split="train")
    with pytest.raises(ValueError):
        build_examples({"dataset": "gsm8k", "engine": "nl", "train_file": str(PYTHON_FILE)}, split="train")


def test_main_configs_carry_the_manuscript_hyperparameters():
    from math_reasoning.train import load_run_config

    for path in sorted(CONFIGS.glob("*.yaml")):
        cfg = load_run_config([str(path)], [])
        t, a = cfg.training, cfg.arix
        assert cfg.method == "arix" and a.use_bank and a.use_contrastive and a.use_partial_reward
        assert (t.epochs, t.learning_rate, t.batch_size, t.group_size) == (10, 1e-6, 32, 64)
        assert (t.clip_eps, t.kl_beta, t.kl_estimator, t.adv_eps, a.tau) == (0.2, 0.04, "k3", 1e-4, 0.2)
        assert (a.alpha_pos, a.partial_scale, a.partial_bias, a.bank_size, a.proj_dim) == (1e-3, 0.1, 0.2, 5, 64)
        assert (a.retrieval, a.bank_update, a.pooling, a.retrieval_start_epoch) == (
            "similarity",
            "redundancy",
            "softmax",
            2,
        )
        dataset = cfg.data["dataset"]
        assert cfg.generation.max_new_tokens == (1024 if dataset == "math" else 512)
        assert cfg.model.name_or_path == ("openai/gpt-oss-20b" if "gptoss" in path.stem else "Qwen/Qwen2.5-7B-Instruct")
        task_from_config(cfg.data)


def test_ablation_overlays():
    from math_reasoning.train import load_run_config, parse_args

    base = str(CONFIGS / "gsm8k_qwen25_7b.yaml")

    def overlay(name: str, *overrides: str):
        return load_run_config([base, str(CONFIGS / "ablations" / f"{name}.yaml")], list(overrides))

    assert not overlay("arix_wo_rb").arix.use_bank
    grpo = overlay("grpo").arix
    assert not (grpo.use_bank or grpo.use_contrastive or grpo.use_partial_reward)
    dr = overlay("dr_grpo").training
    assert not (dr.std_normalize or dr.length_normalize)
    assert overlay("random_retrieval").arix.retrieval == "random"
    assert overlay("fifo_update").arix.bank_update == "fifo"
    assert overlay("random_update").arix.bank_update == "random"
    assert overlay("mean_pooling").arix.pooling == "mean"
    assert overlay("proj32").arix.proj_dim == 32
    assert overlay("k10").arix.bank_size == 10
    assert overlay("g8").training.group_size == 8
    no_partial = overlay("no_partial_reward").arix
    assert not no_partial.use_bank and no_partial.use_contrastive and not no_partial.use_partial_reward
    no_contrastive = overlay("no_contrastive").arix
    assert not no_contrastive.use_bank and not no_contrastive.use_contrastive and no_contrastive.use_partial_reward
    negative = overlay("negative_loss").arix
    assert not negative.use_bank and negative.use_negative and negative.alpha_neg == 5e-4
    assert overlay("k10").output.output_dir == "outputs/math/gsm8k_qwen25_7b-k10"
    assert overlay("k10", "output.output_dir=runs/k10").output.output_dir == "runs/k10"
    assert len(list((CONFIGS / "ablations").glob("*.yaml"))) == 13

    args, overrides = parse_args(["--config", base, "training.seed=2222", "--arix.tau=0.1"])
    assert args.config == [base] and overrides == ["training.seed=2222", "--arix.tau=0.1"]
    assert load_run_config(args.config, overrides).training.seed == 2222


def test_summary_uses_integer_counts():
    from math_reasoning.evaluate import summarize

    records = [{"correct": [True, False, False], "num_correct": 1}, {"correct": [True], "num_correct": 1}]
    summary = summarize(records, diversity=False)
    assert (summary["correct"], summary["total"], summary["num_problems"]) == (2, 4, 2)
    assert summary["accuracy"] == 50.0
    with_diversity = [
        {**records[0], "self_bleu": 0.5, "cosine_distance": 0.1},
        {**records[1], "self_bleu": 1.0, "cosine_distance": 0.3},
    ]
    summary = summarize(with_diversity, diversity=True)
    assert (summary["self_bleu"], summary["mean_pairwise_cosine_distance"]) == pytest.approx((0.75, 0.2))


def _script_generation(monkeypatch, respond):
    import arix.evaluation

    calls = []

    def generate(_model, tokenizer, prompts, *_settings):
        calls.append(len(prompts))
        rollouts = {}
        responses = []
        for prompt in prompts:
            number = rollouts[tuple(prompt)] = rollouts.get(tuple(prompt), -1) + 1
            text = respond(tokenizer.decode(prompt), number)
            responses.append(tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id])
        return responses

    monkeypatch.setattr(arix.evaluation, "generate_batch", generate)
    return calls


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_evaluate_end_to_end_on_cpu(tiny_model_dir, tmp_path):
    from math_reasoning.evaluate import main

    common = ["--model", str(tiny_model_dir), "--dataset", "gsm8k", "--engine", "nl", "--eval_file", str(NL_FILE)]
    common += ["--torch_dtype", "float32", "--max_new_tokens", "4", "--batch_size", "8"]
    greedy_file = tmp_path / "greedy.jsonl"
    summary = main([*common, "--output_file", str(greedy_file)])
    lines = [json.loads(line) for line in greedy_file.read_text(encoding="utf-8").splitlines()]
    assert [line["index"] for line in lines] == list(range(16))
    assert summary["total"] == 16 and summary["correct"] == sum(line["num_correct"] for line in lines)
    assert summary["accuracy"] == 100.0 * summary["correct"] / 16
    assert json.loads(greedy_file.with_suffix(".metrics.json").read_text(encoding="utf-8"))["total"] == 16

    sampled_file = tmp_path / "sampled.jsonl"
    sampled = main(
        [*common, "--num_samples", "2", "--temperature", "1.0", "--diversity", "--output_file", str(sampled_file)]
    )
    assert sampled["total"] == 32
    assert 0.0 <= sampled["self_bleu"] <= 1.0
    assert 0.0 <= sampled["mean_pairwise_cosine_distance"] <= 2.0
    first = json.loads(sampled_file.read_text(encoding="utf-8").splitlines()[0])
    assert len(first["responses"]) == 2 and "cosine_distance" in first


ALWAYS_CORRECT = {0, 3, 9, 15}
SECOND_CORRECT = {1, 2}
WRONG = "Therefore, the answer is: 7"


def test_evaluate_scores_scripted_responses_exactly(tiny_model_dir, tmp_path, monkeypatch):
    import yaml

    from arix.diversity import self_bleu
    from math_reasoning.evaluate import main

    problems = load_reft(NL_FILE)

    def respond(prompt, rollout):
        index = next(i for i, problem in enumerate(problems) if problem.question.strip() in prompt)
        if index in ALWAYS_CORRECT or (rollout == 1 and index in SECOND_CORRECT):
            return f"She sold 48 / 2 = 24 clips.\nTherefore, the answer is: {problems[index].answer}"
        return "I cannot solve this one" if index == 4 else WRONG

    calls = _script_generation(monkeypatch, respond)
    config = tmp_path / "config.yaml"
    data = {"dataset": "gsm8k", "engine": "nl", "test_file": str(NL_FILE)}
    config.write_text(yaml.safe_dump({"data": data, "evaluation": {"batch_size": 6}}), encoding="utf-8")
    common = ["--model", str(tiny_model_dir), "--config", str(config), "--torch_dtype", "float32"]

    greedy = main([*common, "--output_file", str(tmp_path / "greedy.jsonl")])
    records = _read_jsonl(tmp_path / "greedy.jsonl")
    assert calls == [6, 6, 4]
    assert (greedy["correct"], greedy["total"], greedy["accuracy"]) == (4, 16, 25.0)
    assert [record["correct"] for record in records] == [[index in ALWAYS_CORRECT] for index in range(16)]
    assert [record["predictions"][0] for record in records[3:6]] == [42.0, None, 7.0]
    assert (greedy["batch_size"], greedy["seed"], greedy["max_new_tokens"]) == (6, 1111, 512)
    assert "self_bleu" not in greedy and "self_bleu_n" not in greedy

    calls.clear()
    sampling = ["--num_samples", "2", "--temperature", "1.0", "--diversity", "--batch_size", "8"]
    sampled = main([*common, *sampling, "--output_file", str(tmp_path / "sampled.jsonl")])
    records = _read_jsonl(tmp_path / "sampled.jsonl")
    assert calls == [8, 8, 8, 8]
    assert (sampled["correct"], sampled["total"], sampled["accuracy"]) == (10, 32, 31.25)
    assert [record["num_correct"] for record in records] == [
        2 if index in ALWAYS_CORRECT else int(index in SECOND_CORRECT) for index in range(16)
    ]
    for record in records:
        assert record["self_bleu"] == pytest.approx(self_bleu(record["responses"], n=4, tokenizer="regex"))
        if record["index"] in SECOND_CORRECT:
            assert record["self_bleu"] < 0.5 and record["cosine_distance"] > 1e-4
        else:
            assert record["self_bleu"] == pytest.approx(1.0)
            assert record["cosine_distance"] == pytest.approx(0.0, abs=1e-6)
    assert sampled["self_bleu"] == pytest.approx(sum(record["self_bleu"] for record in records) / 16)
    assert sampled["mean_pairwise_cosine_distance"] == pytest.approx(
        sum(record["cosine_distance"] for record in records) / 16
    )
    assert (sampled["batch_size"], sampled["embed_batch_size"]) == (8, 4)
    assert (sampled["self_bleu_n"], sampled["self_bleu_tokenizer"]) == (4, "regex")


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        (["gsm8k_qwen25_7b.yaml", "--engine", "nl"], ("gsm8k", "nl", "data/gsm8k_test_set.json", 512)),
        (["svamp_qwen25_7b.yaml", "--engine", "nl"], ("svamp", "nl", "data/svamp_test_set.json", 512)),
        (["math_qwen25_7b.yaml"], ("math", "nl", "data/MATH/test", 1024)),
        (
            ["math_qwen25_7b.yaml", "--dataset", "aime", "--eval_file", "data/aime_2025.jsonl"],
            ("aime", "nl", "data/aime_2025.jsonl", 1024),
        ),
    ],
)
def test_a_checkpoint_without_config_is_evaluated_with_a_training_config(options, expected, tiny_model_dir):
    import torch

    from arix.embedding import RationaleProjector
    from arix.evaluation import evaluation_generation, load_projector, read_run_config
    from math_reasoning.evaluate import DEFAULT_MAX_NEW_TOKENS, eval_data_config, parse_args

    assert not {"config.yaml", "projector.pt"} & {path.name for path in tiny_model_dir.iterdir()}
    config, *rest = options
    sampling = ["--num_samples", "16", "--temperature", "1.0", "--diversity"]
    args = parse_args(["--model", str(tiny_model_dir), "--config", str(CONFIGS / config), *rest, *sampling])
    cfg = read_run_config(args.model, args.config)
    data = eval_data_config(args, cfg)
    gen = evaluation_generation(args, cfg, DEFAULT_MAX_NEW_TOKENS[data.dataset])
    assert (data.dataset, data.engine, data.test_file, gen.max_new_tokens) == expected
    assert "def solution" not in make_example(_problem(), data.dataset, data.engine, with_reference=False).prompt
    assert cfg.arix.proj_seed == cfg.training.seed == 1111
    rebuilt = load_projector(args.model, None, cfg.arix, hidden_size=32)
    assert torch.equal(rebuilt.linear.weight, RationaleProjector(32, cfg.arix.proj_dim, 1111).linear.weight)


def _problem():
    return MathProblem(uid="p", question="How many?", answer="3")


def test_a_carft_style_checkpoint_runs_with_a_training_config(tiny_model_dir, tmp_path, monkeypatch):
    import yaml

    import arix.evaluation
    from math_reasoning.evaluate import main

    config = yaml.safe_load((CONFIGS / "gsm8k_qwen25_7b.yaml").read_text(encoding="utf-8"))
    config["training"]["seed"] = 2222
    path = tmp_path / "gsm8k_seed2222.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    built = []

    class RecordingProjector(arix.evaluation.RationaleProjector):
        def __init__(self, hidden_size, proj_dim=64, seed=0):
            built.append((hidden_size, proj_dim, seed))
            super().__init__(hidden_size, proj_dim, seed)

    monkeypatch.setattr(arix.evaluation, "RationaleProjector", RecordingProjector)
    _script_generation(monkeypatch, lambda _prompt, rollout: f"{rollout} apples.\nTherefore, the answer is: 7")
    output = tmp_path / "carft.jsonl"
    summary = main(
        ["--model", str(tiny_model_dir), "--config", str(path), "--engine", "nl", "--eval_file", str(NL_FILE)]
        + ["--torch_dtype", "float32", "--num_samples", "2", "--temperature", "1.0", "--diversity"]
        + ["--output_file", str(output)]
    )
    assert (summary["dataset"], summary["engine"], summary["max_new_tokens"], summary["total"]) == (
        "gsm8k",
        "nl",
        512,
        32,
    )
    assert built == [(32, 64, 2222)]
    assert all(record["predictions"] == [7.0, 7.0] for record in _read_jsonl(output))


def test_train_then_evaluate_the_checkpoint(tiny_model_dir, tmp_path):
    from math_reasoning.evaluate import main as evaluate
    from math_reasoning.train import main as train

    run = tmp_path / "run"
    train(
        [
            "--config",
            str(CONFIGS / "gsm8k_qwen25_7b.yaml"),
            f"model.name_or_path={tiny_model_dir}",
            "model.torch_dtype=float32",
            "model.gradient_checkpointing=false",
            f"data.train_file={PYTHON_FILE}",
            "data.exec_workers=2",
            "training.batch_size=2",
            "training.group_size=4",
            "training.micro_batch_size=4",
            "training.generation_batch_size=8",
            "training.max_steps=2",
            "generation.max_new_tokens=8",
            "arix.bank_size=3",
            f"output.output_dir={run}",
            "output.save_every_steps=0",
        ]
    )
    checkpoint = run / "final"
    assert {"config.yaml", "projector.pt", "bank_rank0.json"} <= {path.name for path in checkpoint.iterdir()}
    assert len((run / "train_log.jsonl").read_text(encoding="utf-8").splitlines()) == 2

    output = tmp_path / "eval.jsonl"
    sampling = ["--num_samples", "2", "--temperature", "1.0", "--diversity", "--batch_size", "8"]
    summary = evaluate(
        ["--model", str(checkpoint), "--eval_file", str(PYTHON_FILE), "--output_file", str(output), *sampling]
    )
    assert (summary["dataset"], summary["engine"], summary["max_new_tokens"]) == ("gsm8k", "python", 8)
    assert summary["total"] == 32 and 0 <= summary["correct"] <= 32
    assert "mean_pairwise_cosine_distance" in summary


def test_the_default_output_name_holds_the_dataset_engine_file_and_decoding() -> None:
    from math_reasoning.evaluate import output_name

    assert output_name("gsm8k", "python", "data/gsm8k_test_set.json", 1) == "gsm8k_python_gsm8k_test_set"
    assert output_name("gsm8k", "nl", "data/gsm8k_test_set.json", 16) == "gsm8k_nl_gsm8k_test_set_samples16"
    assert output_name("gsm8k", "nl", "data/gsm8k_test_set.json", 16, 1.0) == "gsm8k_nl_gsm8k_test_set_samples16"
    assert output_name("aime", "nl", Path("data/aime_2025.jsonl"), 1) == "aime_nl_aime_2025"
    assert output_name("aime", "nl", "data/aime_2025.jsonl", 1, 0.6, 0.95) == "aime_nl_aime_2025_t0.6_p0.95"
    assert output_name("gsm8k", "nl", "data/gsm8k_test_set.json", 1, 1.0) == "gsm8k_nl_gsm8k_test_set_t1"


def test_a_sampled_evaluation_keeps_the_greedy_outputs(tiny_model_dir, tmp_path, monkeypatch):
    import shutil

    import yaml

    from math_reasoning.evaluate import main

    checkpoint = tmp_path / "final"
    shutil.copytree(tiny_model_dir, checkpoint)
    data = {"dataset": "gsm8k", "engine": "nl", "test_file": str(NL_FILE)}
    (checkpoint / "config.yaml").write_text(yaml.safe_dump({"data": data}), encoding="utf-8")
    _script_generation(monkeypatch, lambda _prompt, _rollout: WRONG)
    common = ["--model", str(checkpoint), "--torch_dtype", "float32"]
    greedy = main(common)
    output = checkpoint / "eval" / "gsm8k_nl_gsm8k_nl_sample.jsonl"
    written = output.read_bytes(), output.with_suffix(".metrics.json").read_bytes()
    sampled = main([*common, "--temperature", "1.0"])
    assert (greedy["temperature"], sampled["temperature"]) == (0.0, 1.0)
    assert (output.read_bytes(), output.with_suffix(".metrics.json").read_bytes()) == written
    assert sorted(path.name for path in output.parent.iterdir()) == [
        "gsm8k_nl_gsm8k_nl_sample.jsonl",
        "gsm8k_nl_gsm8k_nl_sample.metrics.json",
        "gsm8k_nl_gsm8k_nl_sample_t1.jsonl",
        "gsm8k_nl_gsm8k_nl_sample_t1.metrics.json",
    ]
