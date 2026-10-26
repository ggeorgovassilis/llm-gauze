"""Rule-driven prompt trimming (#153) — unit and validation.

The unit tests pin the four actions, match keys, ordering/terminals,
composition, every safety invariant, determinism and the scan bound. The
validation tests prove an invalid file logs one ``ERROR`` and disables
trimming. That requests are still served in that state, and every end-to-end
proxy behaviour, are asserted in ``test_prompt_trim_integration.py``.

All prompt text here is synthetic — never recorded conversation content.
"""

import copy
import hashlib
import json
import logging
import re
from pathlib import Path

import pytest
from app.remediation.prompt_trim import PromptTrimTransform, validate_rules
from conftest import rules_document, write_rules
from mock_upstream import settings_override

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_RULES = REPO_ROOT / "config" / "prompt_rules.example.json"


def _transform(rules, **kwargs):
    compiled, errors = validate_rules(rules_document(rules), kwargs.get("marker", "... [trimmed]"))
    assert errors == [], errors
    return PromptTrimTransform(compiled, **kwargs)


def _body(*messages, **extra):
    body = {"model": "test", "messages": list(messages)}
    body.update(extra)
    return body


def _contents(body):
    return [message.get("content") for message in body["messages"]]


# --- actions and matching ---------------------------------------------


def test_sub_replaces_all_occurrences_with_backreferences():
    transform = _transform(
        [
            {
                "id": "swap",
                "when": {"role": "system", "content_regex": r"(\w+)-(\w+)"},
                "then": {"action": "sub", "replacement": r"\2/\1"},
            }
        ]
    )
    body = _body({"role": "system", "content": "a-b c-d"})
    out, changes = transform.apply(body)

    assert out["messages"][0]["content"] == "b/a d/c"
    assert changes == [
        {
            "kind": "prompt_trim",
            "rule": "swap",
            "index": 0,
            "role": "system",
            "action": "sub",
            "chars_before": len("a-b c-d"),
            "chars_after": len("b/a d/c"),
        }
    ]


def test_replace_replaces_whole_content_and_is_terminal():
    transform = _transform(
        [
            {
                "id": "replace",
                "when": {"role": "system", "content_regex": "secret"},
                "then": {"action": "replace", "content": "public"},
            },
            {
                "id": "would-otherwise-match",
                "when": {"role": "system", "content_regex": "public"},
                "then": {"action": "sub", "replacement": "later"},
            },
        ]
    )
    out, changes = transform.apply(_body({"role": "system", "content": "secret"}))

    assert out["messages"][0]["content"] == "public"
    assert [change["rule"] for change in changes] == ["replace"]


def test_drop_removes_message_and_is_terminal():
    transform = _transform(
        [
            {
                "id": "drop",
                "when": {"role": "assistant", "content_regex": "^drop me$"},
                "then": {"action": "drop"},
            }
        ]
    )
    body = _body(
        {"role": "user", "content": "keep"},
        {"role": "assistant", "content": "drop me"},
        {"role": "assistant", "content": "keep too"},
    )
    out, changes = transform.apply(body)

    assert _contents(out) == ["keep", "keep too"]
    assert changes == [
        {
            "kind": "prompt_trim",
            "rule": "drop",
            "index": 1,
            "role": "assistant",
            "action": "drop",
            "chars_before": len("drop me"),
            "chars_after": 0,
        }
    ]
    # The input is never mutated.
    assert _contents(body) == ["keep", "drop me", "keep too"]


def test_truncate_bounds_content_and_counts_marker_within_chars():
    transform = _transform(
        [
            {
                "id": "shorten",
                "when": {"role": "system", "content_regex": "^abcdefghijklmnop$"},
                "then": {"action": "truncate", "chars": 10},
            }
        ],
        marker="...",
    )
    out, changes = transform.apply(_body({"role": "system", "content": "abcdefghijklmnop"}))

    assert out["messages"][0]["content"] == "abcdefg..."
    assert len(out["messages"][0]["content"]) == 10
    assert changes[0]["action"] == "truncate"
    assert changes[0]["chars_before"] == 16
    assert changes[0]["chars_after"] == 10


def test_truncate_marker_longer_than_chars_stays_within_bound():
    transform = _transform(
        [
            {
                "id": "shorten",
                "when": {"role": "system", "content_regex": "^abcdef$"},
                "then": {"action": "truncate", "chars": 3},
            }
        ],
        marker="... [trimmed]",
    )
    out, _ = transform.apply(_body({"role": "system", "content": "abcdef"}))
    assert out["messages"][0]["content"] == "..."


@pytest.mark.parametrize(("content", "chars"), [("abc", 100), ("abc", 3)])
def test_truncate_never_grows_content_that_already_fits(content, chars):
    # A plain "cap this preamble" rule is accepted at load time; a message
    # inside the bound (or exactly on it) must be left byte-identical and
    # unrecorded.
    transform = _transform(
        [
            {
                "id": "bounded",
                "when": {"role": "system", "content_regex": r"^abc$"},
                "then": {"action": "truncate", "chars": chars},
            }
        ],
        marker="... [trimmed]",
    )
    body = _body({"role": "system", "content": content})
    out, changes = transform.apply(body)

    assert out is body
    assert changes == []
    assert out["messages"][0]["content"] == content


def test_noop_truncate_does_not_block_a_later_rule():
    # A truncate that leaves the content within the bound is a no-op; rule
    # evaluation must continue with the later rule in file order.
    transform = _transform(
        [
            {
                "id": "no-op-cap",
                "when": {"role": "system", "content_regex": r"^abc$"},
                "then": {"action": "truncate", "chars": 100},
            },
            {
                "id": "rewrite",
                "when": {"role": "system", "content_regex": r"^abc$"},
                "then": {"action": "replace", "content": "xyz"},
            },
        ]
    )
    out, changes = transform.apply(_body({"role": "system", "content": "abc"}))

    assert out["messages"][0]["content"] == "xyz"
    assert [change["rule"] for change in changes] == ["rewrite"]


def test_rules_compose_in_file_order():
    transform = _transform(
        [
            {
                "id": "first",
                "when": {"role": "system", "content_regex": "a"},
                "then": {"action": "sub", "replacement": "b"},
            },
            {
                "id": "second",
                "when": {"role": "system", "content_regex": "b"},
                "then": {"action": "sub", "replacement": "c"},
            },
            {
                "id": "third",
                "when": {"role": "system", "content_regex": "c"},
                "then": {"action": "replace", "content": "d"},
            },
        ]
    )
    out, changes = transform.apply(_body({"role": "system", "content": "a"}))

    assert out["messages"][0]["content"] == "d"
    assert [change["rule"] for change in changes] == ["first", "second", "third"]


def test_sub_and_truncate_compose_on_the_same_message():
    transform = _transform(
        [
            {
                "id": "expand",
                "when": {"role": "system", "content_regex": "^a$"},
                "then": {"action": "sub", "replacement": "bcdefghij"},
            },
            {
                "id": "shorten",
                "when": {"role": "system", "content_regex": "^bcdefghij$"},
                "then": {"action": "truncate", "chars": 5},
            },
        ],
        marker="..",
    )
    out, changes = transform.apply(_body({"role": "system", "content": "a"}))

    assert out["messages"][0]["content"] == "bcd.."
    assert [change["rule"] for change in changes] == ["expand", "shorten"]


def test_role_accepts_a_string_or_a_list():
    transform = _transform(
        [
            {
                "id": "roles",
                "when": {"role": ["system", "assistant"], "content_regex": "x"},
                "then": {"action": "replace", "content": "y"},
            }
        ]
    )
    out, changes = transform.apply(
        _body(
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "x"},
        )
    )

    assert _contents(out) == ["x", "y"]
    assert [change["index"] for change in changes] == [1]


def test_missing_role_matches_any_role():
    transform = _transform(
        [
            {
                "id": "any-role",
                "when": {"content_regex": "x"},
                "then": {"action": "replace", "content": "y"},
            }
        ]
    )
    out, _ = transform.apply(_body({"role": "user", "content": "x"}))
    assert out["messages"][0]["content"] == "y"


def test_content_regex_uses_search_not_fullmatch():
    transform = _transform(
        [
            {
                "id": "search",
                "when": {"role": "system", "content_regex": "mid"},
                "then": {"action": "replace", "content": "hit"},
            }
        ]
    )
    out, _ = transform.apply(_body({"role": "system", "content": "a mid b"}))
    assert out["messages"][0]["content"] == "hit"


def test_min_chars_gates_matching():
    rule = {
        "id": "long-enough",
        "when": {"role": "system", "content_regex": "x", "min_chars": 5},
        "then": {"action": "replace", "content": "y"},
    }
    transform = _transform([rule])
    short, short_changes = transform.apply(_body({"role": "system", "content": "xxxx"}))
    long, long_changes = transform.apply(_body({"role": "system", "content": "xxxxx"}))

    assert short["messages"][0]["content"] == "xxxx"
    assert short_changes == []
    assert long["messages"][0]["content"] == "y"
    assert len(long_changes) == 1


def test_non_string_content_is_skipped_untouched():
    transform = _transform(
        [
            {
                "id": "any",
                "when": {"role": "system", "content_regex": "^SECRET$"},
                "then": {"action": "replace", "content": "changed"},
            }
        ]
    )
    body = _body(
        {"role": "system", "content": None},
        {"role": "system", "content": [{"type": "text", "text": "parts"}]},
        {"role": "system", "content": "SECRET"},
    )
    out, changes = transform.apply(body)

    assert out["messages"][0] == {"role": "system", "content": None}
    assert out["messages"][1]["content"] == [{"type": "text", "text": "parts"}]
    assert out["messages"][2]["content"] == "changed"
    assert [change["index"] for change in changes] == [2]


def test_non_dict_messages_are_kept_untouched():
    transform = _transform(
        [
            {
                "id": "any",
                "when": {"role": "system", "content_regex": "^x$"},
                "then": {"action": "replace", "content": "y"},
            }
        ]
    )
    body = _body("not a message", {"role": "system", "content": "x"})
    out, changes = transform.apply(body)
    assert out["messages"][0] == "not a message"
    assert out["messages"][1]["content"] == "y"
    assert [change["index"] for change in changes] == [1]


def test_change_index_is_the_original_message_index_after_a_drop():
    transform = _transform(
        [
            {
                "id": "drop-first",
                "when": {"role": "assistant", "content_regex": "^drop$"},
                "then": {"action": "drop"},
            },
            {
                "id": "rewrite-second",
                "when": {"role": "assistant", "content_regex": "^keep$"},
                "then": {"action": "replace", "content": "kept"},
            },
        ]
    )
    body = _body(
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "drop"},
        {"role": "assistant", "content": "keep"},
    )
    out, changes = transform.apply(body)

    assert _contents(out) == ["u", "kept"]
    assert [(change["rule"], change["index"]) for change in changes] == [
        ("drop-first", 1),
        ("rewrite-second", 2),
    ]


# --- safety invariants -------------------------------------------------


def test_never_empties_the_messages_list():
    transform = _transform(
        [
            {
                "id": "drop-all",
                "when": {"role": "assistant", "content_regex": "."},
                "then": {"action": "drop"},
            }
        ]
    )
    out, changes = transform.apply(
        _body(
            {"role": "assistant", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "assistant", "content": "three"},
        )
    )

    assert len(out["messages"]) == 1
    assert out["messages"][0]["content"] == "three"
    assert len(changes) == 2


def test_never_drops_the_last_user_message():
    transform = _transform(
        [
            {
                "id": "drop-user",
                "when": {"role": "user", "content_regex": "."},
                "then": {"action": "drop"},
            }
        ]
    )
    out, changes = transform.apply(_body({"role": "user", "content": "the request"}))
    assert out["messages"] == [{"role": "user", "content": "the request"}]
    assert changes == []


def test_earlier_user_messages_may_be_dropped_but_not_the_latest():
    transform = _transform(
        [
            {
                "id": "drop-user",
                "when": {"role": "user", "content_regex": "."},
                "then": {"action": "drop"},
            }
        ]
    )
    out, changes = transform.apply(
        _body(
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "latest"},
        )
    )

    assert _contents(out) == ["reply", "latest"]
    assert [change["index"] for change in changes] == [0]


def test_drop_never_orphans_tool_calls_or_tool_results():
    transform = _transform(
        [
            {
                "id": "drop-pair",
                "when": {"role": ["assistant", "tool"], "content_regex": "."},
                "then": {"action": "drop"},
            }
        ]
    )
    body = _body(
        {
            "role": "assistant",
            "content": "calling",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
        {"role": "user", "content": "latest"},
    )
    out, changes = transform.apply(body)

    assert out == body
    assert changes == []


def test_plain_messages_can_still_be_dropped_next_to_a_tool_pair():
    transform = _transform(
        [
            {
                "id": "drop-assistant",
                "when": {"role": "assistant", "content_regex": "^drop$"},
                "then": {"action": "drop"},
            }
        ]
    )
    body = _body(
        {"role": "assistant", "content": "drop"},
        {
            "role": "assistant",
            "content": "calling",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
        {"role": "user", "content": "latest"},
    )
    out, changes = transform.apply(body)

    assert _contents(out) == ["calling", "result", "latest"]
    assert [change["index"] for change in changes] == [0]


def test_only_messages_content_is_rewritten():
    transform = _transform(
        [
            {
                "id": "replace",
                "when": {"role": "system", "content_regex": "secret"},
                "then": {"action": "replace", "content": "public"},
            }
        ]
    )
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    body = _body(
        {"role": "system", "content": "secret", "name": "kept"},
        {"role": "user", "content": "task", "tool_call_id": "keep"},
        tools=tools,
        functions=[{"name": "f"}],
        tool_choice="auto",
        temperature=0.2,
    )
    before = copy.deepcopy(body)
    out, _ = transform.apply(body)

    assert body == before  # input not mutated
    assert {key: value for key, value in out.items() if key != "messages"} == {
        key: value for key, value in body.items() if key != "messages"
    }
    assert out["messages"][0] == {"role": "system", "content": "public", "name": "kept"}
    assert out["messages"][1] is body["messages"][1]


def test_disabled_no_match_and_non_dict_bodies_are_untouched():
    disabled = PromptTrimTransform([], enabled=False, disabled_reason="disabled")
    body = _body({"role": "system", "content": "x"})
    assert disabled.apply(body) == (body, [])
    assert disabled.process(body).changed is False

    no_match = _transform(
        [
            {
                "id": "no-match",
                "when": {"role": "system", "content_regex": "zzz"},
                "then": {"action": "replace", "content": "y"},
            }
        ]
    )
    out, changes = no_match.apply(body)
    assert out is body
    assert changes == []

    assert no_match.apply({"messages": "not a list"}) == ({"messages": "not a list"}, [])
    assert no_match.apply({"messages": []}) == ({"messages": []}, [])


def test_dry_run_returns_the_original_body_but_records_changes():
    transform = _transform(
        [
            {
                "id": "replace",
                "when": {"role": "system", "content_regex": "secret"},
                "then": {"action": "replace", "content": "public"},
            }
        ],
        dry_run=True,
    )
    body = _body({"role": "system", "content": "secret"})
    result = transform.process(body)

    assert result.body is body
    assert result.changed is False
    assert result.changes[0]["chars_after"] == len("public")
    assert transform.apply(body) == (body, result.changes)


def test_scan_bound_skips_long_messages_and_reports_the_skip():
    transform = _transform(
        [
            {
                "id": "replace",
                "when": {"role": "system", "content_regex": "^x+$"},
                "then": {"action": "replace", "content": "y"},
            }
        ],
        max_scan_chars=5,
    )
    body = _body({"role": "system", "content": "a" * 6})
    result = transform.process(body)

    assert result.body is body
    assert result.changes == []
    assert result.scan_skips == 1


def test_scan_bound_debug_log_has_sizes_only(caplog):
    transform = _transform(
        [
            {
                "id": "replace",
                "when": {"role": "system", "content_regex": "^x+$"},
                "then": {"action": "replace", "content": "y"},
            }
        ],
        max_scan_chars=5,
    )
    secret = "SECRET-PROMPT-TEXT"
    with caplog.at_level(logging.DEBUG, logger="llm_gauze.prompt_trim"):
        transform.apply(_body({"role": "system", "content": secret + "aaaaaa"}))
    assert "chars=24" in caplog.text
    assert secret not in caplog.text


def test_application_debug_log_has_ids_and_sizes_only(caplog):
    transform = _transform(
        [
            {
                "id": "rewrite",
                "when": {"role": "system", "content_regex": "SECRET-PROMPT-TEXT"},
                "then": {"action": "replace", "content": "x"},
            }
        ]
    )
    with caplog.at_level(logging.DEBUG, logger="llm_gauze.prompt_trim"):
        transform.apply(_body({"role": "system", "content": "SECRET-PROMPT-TEXT"}))
    assert "prompt trim rule=rewrite" in caplog.text
    assert "chars_before=18" in caplog.text
    assert "SECRET-PROMPT-TEXT" not in caplog.text


# --- rule validation ---------------------------------------------------


def _valid_rule(**overrides):
    rule = {
        "id": "rule",
        "when": {"role": "system", "content_regex": "foo"},
        "then": {"action": "sub", "replacement": "bar"},
    }
    rule.update(overrides)
    return rule


@pytest.mark.parametrize(
    "document",
    [
        [],
        {},
        {"version": True, "rules": []},
        {"version": 2, "rules": []},
        {"version": 1, "rules": "not a list"},
        {"version": 1, "rules": ["not an object"]},
        rules_document([_valid_rule(id="")]),
        rules_document([_valid_rule(id="same"), _valid_rule(id="same")]),
        rules_document([_valid_rule(then={"action": "unknown"})]),
        rules_document([_valid_rule(when={"role": "system", "content_regex": "("})]),
        rules_document([_valid_rule(when={"role": 1, "content_regex": "x"})]),
        rules_document([_valid_rule(when={"role": ["system", 1], "content_regex": "x"})]),
        rules_document(
            [_valid_rule(when={"role": "system", "content_regex": "x", "min_chars": -1})]
        ),
        rules_document([_valid_rule(then={"action": "replace"})]),
        rules_document([_valid_rule(then={"action": "sub"})]),
        rules_document([_valid_rule(then={"action": "truncate"})]),
        rules_document([_valid_rule(then={"action": "truncate", "chars": True})]),
        rules_document([{"id": "no-when", "then": {"action": "drop"}}]),
        rules_document([{"id": "no-then", "when": {"content_regex": "x"}}]),
        rules_document([_valid_rule(when={"role": "system", "content_regex": 1})]),
        rules_document([_valid_rule(then={"action": "truncate", "chars": -1})]),
        # A rule whose own output still matches its when condition.
        rules_document(
            [
                _valid_rule(
                    when={"role": "system", "content_regex": "foo"},
                    then={"action": "replace", "content": "foo"},
                )
            ]
        ),
        rules_document(
            [
                _valid_rule(
                    when={"role": "system", "content_regex": "foo"},
                    then={"action": "sub", "replacement": "foo"},
                )
            ]
        ),
    ],
)
def test_invalid_documents_are_rejected(document):
    _, errors = validate_rules(document)
    assert errors, document


def test_non_idempotent_rule_without_a_witness_is_rejected():
    # ``(foo)\1`` has a backreference, so no witness can be generated and the
    # replacement-text fallback must catch the non-idempotent output.
    document = rules_document(
        [
            _valid_rule(
                when={"role": "system", "content_regex": r"(foo)\1"},
                then={"action": "sub", "replacement": "foofoo"},
            )
        ]
    )
    rules, errors = validate_rules(document)
    assert rules == []
    assert any("not idempotent" in error for error in errors)


def test_witness_generation_guards_the_private_re_parser_dependency():
    # ``_witness_for`` relies on CPython's private ``re._parser``. This pins
    # that dependency so a Python base-image bump that removes or changes the
    # parser fails here instead of silently accepting a looping rule.
    from app.remediation import prompt_trim

    assert prompt_trim._re_parser is not None  # type: ignore[attr-defined]
    pattern = re.compile(r"(?<=x)y")
    assert prompt_trim._witness_for(pattern) == "xy"

    # The replacement text alone does not match the lookbehind pattern, so
    # only the generated witness can prove the ``sub`` loops.
    document = rules_document(
        [
            _valid_rule(
                when={"role": "system", "content_regex": r"(?<=x)y"},
                then={"action": "sub", "replacement": "y"},
            )
        ]
    )
    rules, errors = validate_rules(document)
    assert rules == []
    assert any("not idempotent" in error for error in errors)


def test_witnessless_rule_with_a_safe_replacement_is_accepted():
    document = rules_document(
        [
            _valid_rule(
                when={"role": "system", "content_regex": r"(foo)\1"},
                then={"action": "sub", "replacement": "bar"},
            )
        ]
    )
    rules, errors = validate_rules(document)
    assert errors == []
    assert [rule.id for rule in rules] == ["rule"]


def test_valid_document_compiles_all_actions():
    document = rules_document(
        [
            _valid_rule(
                id="drop",
                when={"role": "system", "content_regex": "drop"},
                then={"action": "drop"},
            ),
            _valid_rule(
                id="replace",
                when={"role": "system", "content_regex": "replace"},
                then={"action": "replace", "content": "new"},
            ),
            _valid_rule(
                id="sub",
                when={"role": ["system", "assistant"], "content_regex": "sub"},
                then={"action": "sub", "replacement": "new"},
            ),
            _valid_rule(
                id="truncate",
                when={"role": "system", "content_regex": "^truncate$"},
                then={"action": "truncate", "chars": 5},
            ),
        ]
    )
    rules, errors = validate_rules(document, "...")
    assert errors == []
    assert [rule.id for rule in rules] == ["drop", "replace", "sub", "truncate"]
    assert rules[2].roles == ("system", "assistant")


def test_starter_rule_set_is_valid_and_idempotent():
    document = json.loads(EXAMPLE_RULES.read_text(encoding="utf-8"))
    rules, errors = validate_rules(document)
    assert errors == []
    assert [rule.id for rule in rules] == [rule["id"] for rule in document["rules"]]


def test_from_settings_rejects_an_empty_rules_path(caplog):
    with settings_override(prompt_trim_enabled=True, prompt_trim_rules_file=""):
        with caplog.at_level(logging.ERROR, logger="llm_gauze.prompt_trim"):
            transform = PromptTrimTransform.from_settings()
    assert transform.disabled_reason == "invalid_rules"
    assert caplog.text.count("PROMPT_TRIM_RULES_FILE is empty") == 1


def test_from_settings_rejects_a_missing_file(tmp_path, caplog):
    missing = tmp_path / "missing.json"
    with settings_override(prompt_trim_enabled=True, prompt_trim_rules_file=str(missing)):
        with caplog.at_level(logging.ERROR, logger="llm_gauze.prompt_trim"):
            transform = PromptTrimTransform.from_settings()
    assert transform.disabled_reason == "invalid_rules"
    assert caplog.text.count("cannot load prompt-trim rule file") == 1


def test_from_settings_rejects_malformed_json(tmp_path, caplog):
    path = tmp_path / "rules.json"
    path.write_text("{not json", encoding="utf-8")
    with settings_override(prompt_trim_enabled=True, prompt_trim_rules_file=str(path)):
        with caplog.at_level(logging.ERROR, logger="llm_gauze.prompt_trim"):
            transform = PromptTrimTransform.from_settings()
    assert transform.disabled_reason == "invalid_rules"
    assert caplog.text.count("cannot load prompt-trim rule file") == 1


def test_from_settings_loads_a_valid_file(tmp_path):
    path = write_rules(
        tmp_path,
        [
            _valid_rule(
                when={"role": "system", "content_regex": "foo"},
                then={"action": "replace", "content": "bar"},
            )
        ],
    )
    with settings_override(
        prompt_trim_enabled=True,
        prompt_trim_rules_file=str(path),
        prompt_trim_truncate_marker="...",
        prompt_trim_dry_run=True,
        prompt_trim_max_scan_chars=1234,
    ):
        transform = PromptTrimTransform.from_settings()
    assert transform.enabled is True
    assert transform.disabled_reason is None
    assert transform.dry_run is True
    assert transform.max_scan_chars == 1234
    assert transform.marker == "..."
    assert [rule.id for rule in transform.rules] == ["rule"]


def test_from_settings_accepts_and_applies_a_preamble_cap_rule(tmp_path):
    """A plain "cap this preamble at N characters" rule must load and trim.

    ``truncate`` converges in one step, so it is idempotent; the loader must
    not reject it.
    """
    path = write_rules(
        tmp_path,
        [
            {
                "id": "cap-preamble",
                "when": {"role": "system", "content_regex": r"^You are"},
                "then": {"action": "truncate", "chars": 20},
            }
        ],
    )
    with settings_override(
        prompt_trim_enabled=True,
        prompt_trim_rules_file=str(path),
        prompt_trim_truncate_marker="...",
    ):
        transform = PromptTrimTransform.from_settings()

    assert transform.enabled is True
    assert transform.disabled_reason is None
    assert [rule.id for rule in transform.rules] == ["cap-preamble"]

    content = "You are a helpful assistant. " + "x" * 100
    out, changes = transform.apply(_body({"role": "system", "content": content}))

    assert out["messages"][0]["content"] == content[:17] + "..."
    assert changes == [
        {
            "kind": "prompt_trim",
            "rule": "cap-preamble",
            "index": 0,
            "role": "system",
            "action": "truncate",
            "chars_before": len(content),
            "chars_after": 20,
        }
    ]


# --- determinism and the shipped rule set ------------------------------


def _example_transform():
    document = json.loads(EXAMPLE_RULES.read_text(encoding="utf-8"))
    rules, errors = validate_rules(document)
    assert errors == []
    return PromptTrimTransform(rules)


def test_two_runs_are_byte_identical_and_hash_the_same(copilot_body):
    body = copilot_body
    first, first_changes = _example_transform().apply(copy.deepcopy(body))
    second, second_changes = _example_transform().apply(copy.deepcopy(body))

    first_bytes = json.dumps(first).encode("utf-8")
    second_bytes = json.dumps(second).encode("utf-8")
    assert first_bytes == second_bytes
    assert hashlib.sha256(first_bytes).hexdigest() == hashlib.sha256(second_bytes).hexdigest()
    assert first_changes == second_changes


def test_trimmed_prefix_is_stable_across_turns(copilot_body):
    body = copilot_body
    transform = _example_transform()
    first, _ = transform.apply(copy.deepcopy(body))

    later = copy.deepcopy(body)
    later["messages"].append({"role": "user", "content": "a later request"})
    second, _ = transform.apply(later)

    # The system preamble (the cacheable prefix) is byte-identical.
    assert first["messages"][0]["content"] == second["messages"][0]["content"]


def test_example_rules_are_idempotent_and_keep_tools_and_latest_request(copilot_body):
    body = copilot_body
    original = copy.deepcopy(body)
    transform = _example_transform()

    once, _ = transform.apply(copy.deepcopy(body))
    twice, second_changes = transform.apply(once)

    assert twice == once
    assert second_changes == []
    assert once["tools"] == original["tools"]
    assert "<toolUseInstructions>" in once["messages"][0]["content"]
    assert "task_complete" in once["messages"][0]["content"]
    assert once["messages"][2]["content"] == original["messages"][2]["content"]
    assert "Add a unit test for the parser." in once["messages"][2]["content"]
