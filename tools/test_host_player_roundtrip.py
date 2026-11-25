#!/usr/bin/env python3
"""Public host+player round-trip tests for Escrow.

Exercises the real compiled host (`bin/escrow`) and native player (`bin/escrow-player`)
across:
  1. Regular JSON + Markdown ```json fence + real Nov 25 long-notes/optional-defaults
     capture check (completes with genuine scores, model-produced action executed,
     raw text preserved, notes_clipped documented, applied action matches).
  2. Invalid / ambiguous JSON object (surrounding prose / multiple objects) exhausting
     retries -> non-zero exit, NO results.json, player_failure.json with failureKind=rejected,
     private trajectory status=failed with participant_outcomes=null.
  3. Provider failure (HTTP 500, and missing credentials) -> non-zero exit, NO results.json,
     player_failure.json with failureKind=infra, private trajectory status=failed with
     participant_outcomes=null.
  4. Manifest certification contract (native model + every declared player ID seated,
     COWORLD_LLM_ENABLED + COWORLD_LLM_MODEL on native player and game runnable) plus
     a separate explicit offline scripted fixture (no LLM env) -> exit 0, genuine numeric
     results.json.
"""

import http.server
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GAME_BIN = ROOT / "bin" / "escrow"
PLAYER_BIN = ROOT / "bin" / "escrow-player"
MANIFEST = ROOT / "coworld_manifest_template.json"
REAL_TRAJECTORY = Path(
    "/tmp/inkling-suite-native-diag/wave1/escrow-validation-56000/"
    "escrow-validation-56000_s56000_seat0/trajectory.jsonl"
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def clean_env() -> dict[str, str]:
    keep = ("PATH", "HOME", "TMPDIR", "LANG", "LC_CTYPE")
    return {k: os.environ[k] for k in keep if k in os.environ}


def wait_healthz(port: int, proc: subprocess.Popen, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        assert proc.poll() is None, f"game exited early with {proc.returncode}"
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1):
                return
        except OSError:
            time.sleep(0.05)
    raise AssertionError("healthz timeout")


def run_episode(
    work: Path,
    turns: int,
    seed: int,
    player_envs: list[dict[str, str]],
    llm_endpoint: str | None = None,
    capture: bool = True,
) -> tuple[int, Path]:
    port = free_port()
    config = {
        "tokens": [f"tok-{i}" for i in range(4)],
        "players": [{"name": f"Seat{i}"} for i in range(4)],
        "num_agents": 4,
        "seed": seed,
        "turns": turns,
        "talk": True,
        "turnDelayMs": 0,
        "llmTimeoutSeconds": 10,
        "player_connect_timeout_seconds": 15,
    }
    (work / "config.json").write_text(json.dumps(config))
    env = clean_env() | {
        "COGAME_HOST": "127.0.0.1",
        "COGAME_PORT": str(port),
        "COGAME_CONFIG_URI": (work / "config.json").as_uri(),
        "COGAME_RESULTS_URI": (work / "results.json").as_uri(),
        "COGAME_SAVE_REPLAY_URI": (work / "replay.json").as_uri(),
        "COGAME_PLAYER_FAILURE_URI": (work / "player_failure.json").as_uri(),
        "COWORLD_EPISODE_ID": f"roundtrip-{work.name}",
        "COWORLD_GAME_VERSION": "test-local",
        "COWORLD_SOURCE_REVISION": "test-local",
    }
    if capture:
        env["COGAME_SAVE_TRAJECTORY_URI"] = (work / "trajectory.jsonl").as_uri()
    if llm_endpoint:
        env["COWORLD_LLM_ENDPOINT"] = llm_endpoint
        env["COWORLD_LLM_MODEL"] = "local/test-model"

    procs: list[subprocess.Popen] = []
    logs = []
    try:
        glog = (work / "game.log").open("w")
        logs.append(glog)
        game = subprocess.Popen(
            [str(GAME_BIN)], env=env, stdout=glog, stderr=subprocess.STDOUT
        )
        procs.append(game)
        wait_healthz(port, game)
        for slot in range(4):
            penv = (
                clean_env()
                | {
                    "COWORLD_PLAYER_WS_URL": f"ws://127.0.0.1:{port}/player?slot={slot}&token=tok-{slot}"
                }
                | player_envs[slot]
            )
            plog = (work / f"player-{slot}.log").open("w")
            logs.append(plog)
            procs.append(
                subprocess.Popen(
                    [str(PLAYER_BIN)], env=penv, stdout=plog, stderr=subprocess.STDOUT
                )
            )
        rc = game.wait(timeout=45)
        for p in procs[1:]:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.terminate()
                p.wait(timeout=5)
        return rc, work
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.kill()
        for f in logs:
            f.close()


def start_mock_llm(handler_fn):
    calls = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_POST(self):
            length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(length))
            slot = int(self.headers["X-Coworld-Player-Slot"])
            status, text = handler_fn(slot, body, calls)
            call_id = str(uuid.uuid4())
            calls.append({"slot": slot, "status": status, "text": text, "id": call_id})
            if status == 200:
                payload = {
                    "id": f"msg-{call_id}",
                    "model": "local/test-model",
                    "content": [{"type": "text", "text": text}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 10},
                }
            else:
                payload = {"error": {"message": text}}
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("X-Softmax-Llm-Call-Id", call_id)
            self.end_headers()
            self.wfile.write(raw)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}", calls


def test_regular_json_fence_and_clipped_notes(base: Path) -> None:
    """Case 1: Regular JSON, ```json fence, and >600-char notes with omitted offer/give."""
    long_notes = (
        "Turn planning with detailed arithmetic: "
        + "holding surplus ore for commission conversion; " * 25
    )
    assert len(long_notes) > 600

    def responder(slot, body, calls):
        user = body["messages"][0]["content"]
        m = re.search(r"^Turn (\d+) of (\d+)", user)
        turn = int(m.group(1))
        signable = user.split("SIGNABLE NOW", 1)[1].split("YOUR OPEN OFFERS", 1)[0]
        affordable = re.findall(r"^\s+(C\d+) from .*— AFFORDABLE$", signable, re.MULTILINE)
        if turn == 0:
            # Bare regular JSON with real model action (say + notes)
            return 200, json.dumps({"say": f"bare-json-turn-{turn}", "notes": "short note"})
        if turn == 1:
            # ```json fenced object with >600-char notes and omitted optional offer/give
            action = {
                "sign": affordable[:2],
                "say": "fenced-long-notes-turn-1",
                "notes": long_notes,
            }
            return 200, "```json\n" + json.dumps(action) + "\n```"
        if turn == 2:
            # Plain ``` fence with optional fields omitted
            return 200, "```\n" + json.dumps({"sign": affordable[:1], "notes": long_notes}) + "\n```"
        return 200, json.dumps({"sign": affordable[:2], "say": "closing"})

    server, endpoint, calls = start_mock_llm(responder)
    try:
        work = base / "regular-json-fence"
        work.mkdir()
        rc, _ = run_episode(
            work,
            turns=4,
            seed=11,
            player_envs=[
                {"COWORLD_LLM_ENABLED": "true"},
                {"PLAYER_SCRIPTED": "trader"},
                {"PLAYER_SCRIPTED": "trader"},
                {"PLAYER_SCRIPTED": "hoarder"},
            ],
            llm_endpoint=endpoint,
            capture=True,
        )
        assert rc == 0, (work / "game.log").read_text()
        assert not (work / "player_failure.json").exists()
        results = json.loads((work / "results.json").read_text())
        assert results["reason"] == "complete"
        assert results["turns"] == 4
        assert len(results["scores"]) == 4 and all(isinstance(s, int) for s in results["scores"])
        assert len(calls) == 4, f"expected 4 model calls for seat 0, got {len(calls)}"

        # Check native trajectory: all 4 seat-0 model decisions accepted, including clipped notes!
        rows = [json.loads(l) for l in (work / "trajectory.jsonl").read_text().splitlines()]
        seat0_rows = [r for r in rows if r["event_type"] == "decision" and r["seat"] == "0"]
        assert len(seat0_rows) == 4
        for idx, row in enumerate(seat0_rows):
            assert row["action_status"] == "accepted", (
                f"turn {idx} rejected: {row['attempts'][-1].get('rejection_reason')}"
            )
            assert row["selected_attempt_id"] is not None
        # Verify raw text preserved & notes_clipped documented on turn 1
        t1 = seat0_rows[1]
        assert long_notes in t1["attempts"][-1]["response"]
        assert len(t1["executed_action"]["notes"]) == 600
        assert t1["executed_action"]["notes"].endswith("…")
        assert t1["executed_action"]["offer"] == ""
        assert t1["executed_action"]["give"] == []
        assert t1["observation"]["native_execution"]["notes_clipped"] is True
        assert rows[-1]["status"] == "completed"
        print("PASS case 1: regular JSON, ```json/plain fences, clipped notes + optional defaults")
    finally:
        server.shutdown()


def test_invalid_ambiguous_object_fails_unscored(base: Path) -> None:
    """Case 2: Invalid/ambiguous object (two JSON objects / prose) fails unscored."""
    def responder(slot, body, calls):
        return 200, '{"say": "first"} {"say": "second"}'

    server, endpoint, _ = start_mock_llm(responder)
    try:
        work = base / "invalid-ambiguous"
        work.mkdir()
        rc, _ = run_episode(
            work,
            turns=4,
            seed=11,
            player_envs=[
                {"COWORLD_LLM_ENABLED": "true"},
                {"PLAYER_SCRIPTED": "trader"},
                {"PLAYER_SCRIPTED": "trader"},
                {"PLAYER_SCRIPTED": "hoarder"},
            ],
            llm_endpoint=endpoint,
            capture=True,
        )
        assert rc != 0, "ambiguous object must fail episode"
        assert not (work / "results.json").exists(), "must not write numeric results on failure"
        failure = json.loads((work / "player_failure.json").read_text())
        assert failure["status"] == "failed"
        assert failure["failureKind"] == "rejected"
        assert failure["seat"] == 0
        rows = [json.loads(l) for l in (work / "trajectory.jsonl").read_text().splitlines()]
        assert rows[-1]["status"] == "failed"
        assert rows[-1]["participant_outcomes"] is None
        print("PASS case 2: invalid/ambiguous JSON object terminates unscored (failureKind=rejected)")
    finally:
        server.shutdown()


def test_provider_failure_fails_unscored(base: Path) -> None:
    """Case 3a: Provider HTTP 500 failure terminates unscored with failureKind=infra."""
    def responder(slot, body, calls):
        return 500, "upstream provider error"

    server, endpoint, _ = start_mock_llm(responder)
    try:
        work = base / "provider-http-500"
        work.mkdir()
        rc, _ = run_episode(
            work,
            turns=4,
            seed=11,
            player_envs=[
                {"COWORLD_LLM_ENABLED": "true"},
                {"PLAYER_SCRIPTED": "trader"},
                {"PLAYER_SCRIPTED": "trader"},
                {"PLAYER_SCRIPTED": "hoarder"},
            ],
            llm_endpoint=endpoint,
            capture=True,
        )
        assert rc != 0, "provider 500 must fail episode"
        assert not (work / "results.json").exists(), "must not write numeric results on infra failure"
        failure = json.loads((work / "player_failure.json").read_text())
        assert failure["status"] == "failed"
        assert failure["failureKind"] == "infra"
        assert failure["seat"] == 0
        rows = [json.loads(l) for l in (work / "trajectory.jsonl").read_text().splitlines()]
        assert rows[-1]["status"] == "failed"
        assert rows[-1]["participant_outcomes"] is None
        print("PASS case 3a: provider HTTP 500 terminates unscored (failureKind=infra)")
    finally:
        server.shutdown()

    """Case 3b: Missing LLM credentials for model player terminates unscored (failureKind=infra)."""
    work = base / "missing-credentials"
    work.mkdir()
    rc, _ = run_episode(
        work,
        turns=4,
        seed=11,
        player_envs=[
            {"COWORLD_LLM_ENABLED": "true"},
            {"PLAYER_SCRIPTED": "trader"},
            {"PLAYER_SCRIPTED": "trader"},
            {"PLAYER_SCRIPTED": "hoarder"},
        ],
        llm_endpoint=None,
        capture=True,
    )
    assert rc != 0, "missing credentials for model player must fail episode"
    assert not (work / "results.json").exists(), "must not write numeric results when creds missing"
    failure = json.loads((work / "player_failure.json").read_text())
    assert failure["status"] == "failed"
    assert failure["failureKind"] == "infra"
    assert failure["seat"] == 0
    rows = [json.loads(l) for l in (work / "trajectory.jsonl").read_text().splitlines()]
    assert rows[-1]["status"] == "failed"
    assert rows[-1]["participant_outcomes"] is None
    print("PASS case 3b: missing LLM credentials terminates unscored (failureKind=infra)")


def test_manifest_certification_and_offline_scripted(base: Path) -> None:
    """Case 4: Verify manifest release certification seats every declared ID, and run an explicit offline scripted fixture."""
    manifest = json.loads(MANIFEST.read_text())
    players_by_id = {p["id"]: p for p in manifest["player"]}

    # Game runnable performs server-side seat inference, so it must declare LLM env.
    game_env = manifest["game"]["runnable"].get("env", {})
    assert game_env.get("COWORLD_LLM_ENABLED") == "true"
    assert game_env.get("COWORLD_LLM_MODEL"), "game.runnable.env must set default COWORLD_LLM_MODEL"

    # First player in manifest must be the native model player with COWORLD_LLM_ENABLED=true,
    # explicit default COWORLD_LLM_MODEL, and no PLAYER_SCRIPTED.
    first_player = manifest["player"][0]
    assert first_player["id"] == "escrow-player"
    assert first_player["env"].get("COWORLD_LLM_ENABLED") == "true"
    assert first_player["env"].get("COWORLD_LLM_MODEL"), "native player must set default COWORLD_LLM_MODEL"
    assert "PLAYER_SCRIPTED" not in first_player["env"]

    # Release certification fixture must seat every declared manifest player ID.
    cert = manifest["certification"]
    assert len(cert["players"]) == cert["game_config"]["num_agents"] == 4
    seated_ids = {seat["player_id"] for seat in cert["players"]}
    declared_ids = set(players_by_id.keys())
    assert seated_ids == declared_ids, (
        f"certification.players must seat every declared ID {declared_ids}, got {seated_ids}"
    )

    # Separate explicit offline scripted test fixture (no LLM env).
    scripted_ids = ["escrow-trader", "escrow-trader", "escrow-trader", "escrow-hoarder"]
    scripted_envs = []
    for pid in scripted_ids:
        entry = players_by_id[pid]
        assert "PLAYER_SCRIPTED" in entry.get("env", {}), f"{pid} must be explicitly scripted"
        scripted_envs.append(dict(entry["env"]))

    work = base / "offline-scripted"
    work.mkdir()
    rc, _ = run_episode(
        work,
        turns=cert["game_config"]["turns"],
        seed=cert["game_config"]["seed"],
        player_envs=scripted_envs,
        llm_endpoint=None,
        capture=False,
    )
    assert rc == 0, (work / "game.log").read_text()
    assert not (work / "player_failure.json").exists()
    results = json.loads((work / "results.json").read_text())
    assert results["reason"] == "complete"
    assert results["turns"] == cert["game_config"]["turns"]
    assert len(results["scores"]) == 4 and all(isinstance(s, int) for s in results["scores"])
    print("PASS case 4: manifest certification contract valid + explicit offline scripted fixture completes")


def test_nov25_real_trajectory_replay(base: Path) -> None:
    """Case 5: Replay the exact 6 real Nov 25 seat-0 model responses through the live host."""
    if not REAL_TRAJECTORY.is_file():
        print("SKIP case 5: real Nov 25 trajectory not present on this host")
        return
    src_rows = [json.loads(l) for l in REAL_TRAJECTORY.read_text().splitlines()]
    seat0_src = [r for r in src_rows if r["event_type"] == "decision" and r["seat"] == "0"]
    assert len(seat0_src) == 6
    real_responses = [r["attempts"][-1]["response"] for r in seat0_src]

    def responder(slot, body, calls):
        user = body["messages"][0]["content"]
        m = re.search(r"^Turn (\d+) of (\d+)", user)
        turn = int(m.group(1))
        return 200, real_responses[turn]

    server, endpoint, calls = start_mock_llm(responder)
    try:
        work = base / "nov25-exact-replay"
        work.mkdir()
        rc, _ = run_episode(
            work,
            turns=6,
            seed=56000,
            player_envs=[
                {"COWORLD_LLM_ENABLED": "true"},
                {"PLAYER_SCRIPTED": "trader"},
                {"PLAYER_SCRIPTED": "trader"},
                {"PLAYER_SCRIPTED": "trader"},
            ],
            llm_endpoint=endpoint,
            capture=True,
        )
        assert rc == 0, (work / "game.log").read_text()
        assert not (work / "player_failure.json").exists()
        results = json.loads((work / "results.json").read_text())
        assert results["reason"] == "complete"
        assert results["turns"] == 6
        rows = [json.loads(l) for l in (work / "trajectory.jsonl").read_text().splitlines()]
        seat0_new = [r for r in rows if r["event_type"] == "decision" and r["seat"] == "0"]
        assert len(seat0_new) == 6
        for turn, (old_row, new_row) in enumerate(zip(seat0_src, seat0_new)):
            assert new_row["action_status"] == "accepted", (
                f"Nov 25 turn {turn} still rejected: {new_row['attempts'][-1].get('rejection_reason')}"
            )
            assert new_row["selected_attempt_id"] is not None
            assert new_row["attempts"][-1]["response"] == real_responses[turn]
            # Applied game action matches the exact original applied action (including defaults + clipped notes)
            assert new_row["executed_action"] == old_row["executed_action"]
            if turn in (3, 4, 5):
                assert new_row["observation"]["native_execution"]["notes_clipped"] is True
                assert len(new_row["executed_action"]["notes"]) == 600
                assert new_row["executed_action"]["notes"].endswith("…")
        assert rows[-1]["status"] == "completed"
        print(
            f"PASS case 5: exact Nov 25 6-turn replay (seed=56000) — all 6/6 seat-0 turns accepted "
            f"(turns 3,4,5 notes_clipped=True), scores={results['scores']}"
        )
    finally:
        server.shutdown()


def main() -> None:
    assert GAME_BIN.is_file(), f"missing {GAME_BIN}; build bin/escrow first"
    assert PLAYER_BIN.is_file(), f"missing {PLAYER_BIN}; build bin/escrow-player first"
    with tempfile.TemporaryDirectory(prefix="escrow-roundtrip-") as td:
        base = Path(td)
        test_regular_json_fence_and_clipped_notes(base)
        test_invalid_ambiguous_object_fails_unscored(base)
        test_provider_failure_fails_unscored(base)
        test_manifest_certification_and_offline_scripted(base)
        test_nov25_real_trajectory_replay(base)


if __name__ == "__main__":
    main()
