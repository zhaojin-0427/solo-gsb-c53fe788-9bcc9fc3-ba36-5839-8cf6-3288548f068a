"""End-to-end tests for the catalog alert orchestration module."""
from __future__ import annotations

import threading
from datetime import timedelta

from tests.test_catalog import (  # noqa: E402
    BASE_DT,
    _event_window,
    _iso,
    _process,
)

DAY = timedelta(days=1)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _process_at(client, prefix, start, **kwargs):
    """One completed run whose event origin is ``start`` + origin_offset."""
    return _process(client, prefix, 0, start=start, **kwargs)


def _compile_window(client, start, end, **overrides):
    body = {
        "start_time": _iso(start),
        "end_time": _iso(end),
        "radius_km": 50.0,
        "origin_tolerance_s": 5.0,
        "depth_tolerance_km": 15.0,
        "min_quality": "C",
        "min_sources": 1,
        "min_shared_stations": 1,
    }
    body.update(overrides)
    resp = client.post("/v1/catalog/compilations", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _publish_entry(client, entry):
    resp = client.post(
        f"/v1/catalog/entries/{entry['entry_id']}/publish",
        json={"expected_revision": 1},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _published_compilation(client, start, end, **overrides):
    comp = _compile_window(client, start, end, **overrides)
    for entry in comp["entries"]:
        _publish_entry(client, entry)
    return comp


def _make_rule(client, name="rule", version="1.0", publish=True, **conditions):
    body = {"name": name, "version": version}
    body.update(conditions)
    resp = client.post("/v1/alert-rules", json=body)
    assert resp.status_code in (200, 201), resp.text
    rule = resp.json()
    if not publish:
        return rule
    pub = client.post(
        f"/v1/alert-rules/{rule['rule_id']}/publish",
        json={"expected_revision": 1},
    )
    assert pub.status_code == 200, pub.text
    return pub.json()


def _evaluate(client, compilation_id, rule_ids=None):
    body = {"compilation_id": compilation_id}
    if rule_ids is not None:
        body["rule_ids"] = rule_ids
    return client.post("/v1/alerts/evaluations", json=body)


def _three_quakes(client, prefix="Q", gap_s=600.0):
    """Three distinct earthquakes, ``gap_s`` apart, all near (35.05, -117.95)."""
    starts = [BASE_DT + timedelta(seconds=gap_s * i) for i in range(3)]
    for i, start in enumerate(starts):
        _process_at(client, f"{prefix}{i}", start)
    comp = _published_compilation(client, BASE_DT - DAY, BASE_DT + DAY)
    assert comp["entry_count"] == 3, comp
    return comp


# --------------------------------------------------------------------------- #
# rule creation: immutable versions, idempotency, validation
# --------------------------------------------------------------------------- #
def test_rule_create_idempotent_and_conflict(client):
    body = {
        "name": "strong-local", "version": "1.0",
        "min_quality": "B", "min_stations": 6, "max_rms_s": 0.5,
        "merge_interval_s": 600.0, "label": "strong local events",
    }
    r1 = client.post("/v1/alert-rules", json=body)
    assert r1.status_code == 201, r1.text
    rule = r1.json()
    assert rule["created"] is True
    assert rule["status"] == "draft"
    assert rule["revision"] == 1
    assert rule["conditions"]["min_quality"] == "B"
    assert rule["conditions"]["min_stations"] == 6
    assert rule["conditions"]["max_rms_s"] == 0.5
    assert rule["merge_interval_s"] == 600.0

    # identical content is an idempotent replay
    r2 = client.post("/v1/alert-rules", json=body)
    assert r2.status_code == 200
    assert r2.json()["created"] is False
    assert r2.json()["rule_id"] == rule["rule_id"]

    # same name+version with different content is a hard conflict
    changed = client.post(
        "/v1/alert-rules", json={**body, "min_stations": 8}
    )
    assert changed.status_code == 409
    assert changed.json()["detail"]["code"] == "alert_rule_version_conflict"


def test_rule_create_validation(client):
    one_sided = client.post(
        "/v1/alert-rules",
        json={"name": "r", "version": "1.0",
              "window_start": _iso(BASE_DT)},
    )
    assert one_sided.status_code == 422
    assert one_sided.json()["detail"]["code"] == "invalid_rule_time_window"

    backwards = client.post(
        "/v1/alert-rules",
        json={"name": "r", "version": "1.0",
              "window_start": _iso(BASE_DT + DAY),
              "window_end": _iso(BASE_DT)},
    )
    assert backwards.status_code == 422
    assert backwards.json()["detail"]["code"] == "invalid_rule_time_window"

    partial_region = client.post(
        "/v1/alert-rules",
        json={"name": "r", "version": "1.0",
              "center_latitude": 35.0, "center_longitude": -118.0},
    )
    assert partial_region.status_code == 422
    assert partial_region.json()["detail"]["code"] == "invalid_spatial_filter"


def test_rule_publish_and_retire_cas(client):
    rule = _make_rule(client, publish=False)
    rid = rule["rule_id"]

    # draft cannot retire
    retire_draft = client.post(
        f"/v1/alert-rules/{rid}/retire", json={"expected_revision": 1}
    )
    assert retire_draft.status_code == 409
    assert retire_draft.json()["detail"]["code"] == "alert_rule_not_published"

    # stale revision is rejected without changing state
    stale = client.post(
        f"/v1/alert-rules/{rid}/publish", json={"expected_revision": 99}
    )
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "revision_conflict"

    ok = client.post(
        f"/v1/alert-rules/{rid}/publish", json={"expected_revision": 1}
    )
    assert ok.status_code == 200, ok.text
    published = ok.json()
    assert published["status"] == "published"
    assert published["revision"] == 2
    assert published["published_at"] is not None

    # repeat publish is an idempotent no-op (no revision bump)
    again = client.post(
        f"/v1/alert-rules/{rid}/publish", json={"expected_revision": 99}
    )
    assert again.status_code == 200
    assert again.json()["revision"] == 2

    # retire with CAS
    bad = client.post(
        f"/v1/alert-rules/{rid}/retire", json={"expected_revision": 1}
    )
    assert bad.status_code == 409
    assert bad.json()["detail"]["code"] == "revision_conflict"
    retired = client.post(
        f"/v1/alert-rules/{rid}/retire", json={"expected_revision": 2}
    )
    assert retired.status_code == 200
    assert retired.json()["status"] == "retired"
    assert retired.json()["revision"] == 3
    assert retired.json()["retired_at"] is not None

    # a retired rule can neither be published nor re-retired into changes
    repub = client.post(
        f"/v1/alert-rules/{rid}/publish", json={"expected_revision": 3}
    )
    assert repub.status_code == 409
    assert repub.json()["detail"]["code"] == "alert_rule_retired"
    again = client.post(
        f"/v1/alert-rules/{rid}/retire", json={"expected_revision": 3}
    )
    assert again.status_code == 200
    assert again.json()["revision"] == 3


def test_rule_lookup_and_list(client):
    rule = _make_rule(client, name="listed", version="1.0")
    got = client.get(f"/v1/alert-rules/{rule['rule_id']}")
    assert got.status_code == 200
    assert got.json()["name"] == "listed"

    missing = client.get("/v1/alert-rules/deadbeef")
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "alert_rule_not_found"

    listed = client.get("/v1/alert-rules", params={"name": "listed"}).json()
    assert [r["rule_id"] for r in listed["rules"]] == [rule["rule_id"]]
    published = client.get(
        "/v1/alert-rules", params={"status": "published"}
    ).json()
    assert rule["rule_id"] in {r["rule_id"] for r in published["rules"]}
    drafts = client.get("/v1/alert-rules", params={"status": "draft"}).json()
    assert rule["rule_id"] not in {r["rule_id"] for r in drafts["rules"]}


# --------------------------------------------------------------------------- #
# evaluation gating: published rules and published entries only
# --------------------------------------------------------------------------- #
def test_evaluation_requires_published_rules_and_known_scope(client):
    _process_at(client, "G0", BASE_DT)
    comp = _compile_window(client, BASE_DT - DAY, BASE_DT + DAY)

    missing_comp = _evaluate(client, "deadbeef")
    assert missing_comp.status_code == 404
    assert missing_comp.json()["detail"]["code"] == "catalog_compilation_not_found"

    missing_rule = _evaluate(client, comp["compilation_id"], rule_ids=["deadbeef"])
    assert missing_rule.status_code == 404
    assert missing_rule.json()["detail"]["code"] == "alert_rule_not_found"

    draft = _make_rule(client, name="drafty", publish=False)
    not_published = _evaluate(
        client, comp["compilation_id"], rule_ids=[draft["rule_id"]]
    )
    assert not_published.status_code == 409
    assert not_published.json()["detail"]["code"] == "alert_rule_not_published"

    retired = _make_rule(client, name="old", version="1.0")
    client.post(
        f"/v1/alert-rules/{retired['rule_id']}/retire",
        json={"expected_revision": 2},
    )
    gone = _evaluate(client, comp["compilation_id"], rule_ids=[retired["rule_id"]])
    assert gone.status_code == 409
    assert gone.json()["detail"]["code"] == "alert_rule_retired"


def test_only_published_entries_are_evaluated(client):
    _process_at(client, "D0", BASE_DT)
    comp = _compile_window(client, BASE_DT - DAY, BASE_DT + DAY)
    assert comp["entries"][0]["status"] == "draft"
    _make_rule(client, name="allq")

    # draft entries do not alert
    first = _evaluate(client, comp["compilation_id"])
    assert first.status_code == 201, first.text
    assert first.json()["created"] is True
    assert first.json()["entry_count"] == 0
    assert first.json()["new_hit_count"] == 0
    assert first.json()["hits"] == []

    # publishing the entry makes it visible to an idempotent re-evaluation
    _publish_entry(client, comp["entries"][0])
    second = _evaluate(client, comp["compilation_id"])
    assert second.status_code == 200
    assert second.json()["created"] is False
    assert second.json()["evaluation_id"] == first.json()["evaluation_id"]
    assert second.json()["entry_count"] == 1
    assert second.json()["new_hit_count"] == 1
    assert second.json()["evaluation_count"] == 2  # create pass + one re-pass
    hit = second.json()["hits"][0]
    assert hit["entry_id"] == comp["entries"][0]["entry_id"]
    assert hit["disposition"] == "opened"
    assert hit["reason"] is None


def test_rule_conditions_are_conjunctive(client):
    # one A-grade 8-station event near (35.05, -117.95)
    _process_at(client, "C0", BASE_DT)
    comp = _published_compilation(client, BASE_DT - DAY, BASE_DT + DAY)
    cid = comp["compilation_id"]

    matching = _make_rule(
        client, name="cond-match", min_quality="A", min_stations=8,
        max_rms_s=1.0, window_start=_iso(BASE_DT - timedelta(seconds=1)),
        window_end=_iso(BASE_DT + timedelta(seconds=120)),
        center_latitude=35.05, center_longitude=-117.95,
        region_radius_km=50.0,
    )
    ev = _evaluate(client, cid, rule_ids=[matching["rule_id"]])
    assert ev.status_code == 201, ev.text
    assert ev.json()["new_hit_count"] == 1

    # each violated condition independently suppresses the match
    for i, cond in enumerate([
        {"min_quality": "A", "min_stations": 9},          # station count
        {"max_rms_s": 0.0001},                            # rms
        {"window_start": _iso(BASE_DT + timedelta(hours=1)),
         "window_end": _iso(BASE_DT + timedelta(hours=2))},  # time window
        {"center_latitude": 36.5, "center_longitude": -117.95,
         "region_radius_km": 50.0},                       # region
    ]):
        rule = _make_rule(client, name=f"cond-miss-{i}", **cond)
        ev = _evaluate(client, cid, rule_ids=[rule["rule_id"]])
        assert ev.status_code == 201, ev.text
        assert ev.json()["new_hit_count"] == 0, cond


# --------------------------------------------------------------------------- #
# deterministic hits, merging, suppression and escalation
# --------------------------------------------------------------------------- #
def test_consecutive_hits_merge_by_rule_interval(client):
    comp = _three_quakes(client, gap_s=600.0)
    cid = comp["compilation_id"]
    wide = _make_rule(client, name="wide", merge_interval_s=900.0)
    narrow = _make_rule(client, name="narrow", merge_interval_s=300.0)

    ev = _evaluate(client, cid)
    assert ev.status_code == 201, ev.text
    data = ev.json()
    assert data["new_hit_count"] == 6  # 3 entries x 2 rules

    alerts = client.get("/v1/alerts").json()["alerts"]
    by_rule = {}
    for a in alerts:
        by_rule.setdefault(a["rule_name"], []).append(a)
    # 600 s gaps fit the 900 s interval: one alert with three hits
    assert len(by_rule["wide"]) == 1
    assert by_rule["wide"][0]["hit_count"] == 3
    # 600 s gaps exceed the 300 s interval: three independent alerts
    assert len(by_rule["narrow"]) == 3
    assert all(a["hit_count"] == 1 for a in by_rule["narrow"])

    detail = client.get(f"/v1/alerts/{by_rule['wide'][0]['alert_id']}").json()
    dispositions = [h["disposition"] for h in detail["hits"]]
    assert dispositions == ["opened", "merged", "merged"]
    reasons = [h["reason"] for h in detail["hits"]]
    assert reasons == [None, "within_merge_interval", "within_merge_interval"]
    kinds = [e["kind"] for e in detail["events"]]
    assert kinds == ["opened", "merged", "merged"]
    assert detail["status"] == "open"
    assert detail["best_quality"] == "A"
    assert detail["max_station_count"] == 8


def test_merging_spans_compilations(client):
    _process_at(client, "X0", BASE_DT)
    _process_at(client, "X1", BASE_DT + timedelta(seconds=600))
    _make_rule(client, name="xrun", merge_interval_s=900.0)

    comp1 = _published_compilation(
        client, BASE_DT - DAY, BASE_DT + timedelta(seconds=300)
    )
    comp2 = _published_compilation(
        client, BASE_DT + timedelta(seconds=300), BASE_DT + DAY
    )
    assert comp1["entry_count"] == 1 and comp2["entry_count"] == 1

    ev1 = _evaluate(client, comp1["compilation_id"])
    assert ev1.status_code == 201
    alert_id = ev1.json()["hits"][0]["alert_id"]

    ev2 = _evaluate(client, comp2["compilation_id"])
    assert ev2.status_code == 201
    hit2 = ev2.json()["hits"][0]
    # the second compilation's hit merges into the first alert (600 <= 900)
    assert hit2["alert_id"] == alert_id
    assert hit2["disposition"] == "merged"
    assert hit2["reason"] == "within_merge_interval"
    detail = client.get(f"/v1/alerts/{alert_id}").json()
    assert detail["hit_count"] == 2
    assert {h["compilation_id"] for h in detail["hits"]} == {
        comp1["compilation_id"], comp2["compilation_id"]
    }


def test_evaluation_is_idempotent_and_deterministic(client):
    # four quakes 600 s apart in one compilation; only three published at first
    starts = [BASE_DT + timedelta(seconds=600 * i) for i in range(4)]
    for i, start in enumerate(starts):
        _process_at(client, f"I{i}", start)
    comp = _compile_window(client, BASE_DT - DAY, BASE_DT + DAY)
    cid = comp["compilation_id"]
    assert comp["entry_count"] == 4
    _make_rule(client, name="idem", merge_interval_s=900.0)
    for entry in comp["entries"][:3]:
        _publish_entry(client, entry)

    first = _evaluate(client, cid)
    assert first.status_code == 201
    f = first.json()
    assert f["created"] is True
    assert f["new_hit_count"] == 3

    second = _evaluate(client, cid)
    assert second.status_code == 200
    s = second.json()
    assert s["created"] is False
    assert s["evaluation_id"] == f["evaluation_id"]
    assert s["new_hit_count"] == 0
    assert s["hit_count"] == 3
    assert s["matched_count"] == 3
    # identical hit identities and dispositions on re-evaluation
    assert [h["hit_id"] for h in s["hits"]] == [h["hit_id"] for h in f["hits"]]
    assert [h["disposition"] for h in s["hits"]] == [
        h["disposition"] for h in f["hits"]
    ]

    # a newly published entry is picked up by the next idempotent re-pass
    _publish_entry(client, comp["entries"][3])
    third = _evaluate(client, cid)
    assert third.status_code == 200
    t = third.json()
    assert t["new_hit_count"] == 1
    assert t["hit_count"] == 4
    assert t["evaluation_count"] == 3
    # the new hit merges into the same alert (gap 600 s <= 900 s)
    alert_ids = {h["alert_id"] for h in t["hits"]}
    assert len(alert_ids) == 1


def test_escalation_after_acknowledge_and_resolve(client):
    starts = [BASE_DT + timedelta(seconds=600 * i) for i in range(3)]
    for i, start in enumerate(starts):
        _process_at(client, f"E{i}", start)
    comp = _compile_window(client, BASE_DT - DAY, BASE_DT + DAY)
    cid = comp["compilation_id"]
    _make_rule(client, name="esc", merge_interval_s=900.0)

    _publish_entry(client, comp["entries"][0])
    ev = _evaluate(client, cid)
    alert_id = ev.json()["hits"][0]["alert_id"]

    # acknowledge, then a new hit escalates the alert back to open
    ack = client.post(f"/v1/alerts/{alert_id}/acknowledge", json={})
    assert ack.status_code == 200
    assert ack.json()["status"] == "acknowledged"

    _publish_entry(client, comp["entries"][1])
    ev2 = _evaluate(client, cid)
    hit2 = next(
        h for h in ev2.json()["hits"]
        if h["entry_id"] == comp["entries"][1]["entry_id"]
    )
    assert hit2["disposition"] == "escalated"
    assert hit2["reason"] == "new_hit_after_acknowledge"
    detail = client.get(f"/v1/alerts/{alert_id}").json()
    assert detail["status"] == "open"

    # resolve, then the next hit escalates again
    res = client.post(f"/v1/alerts/{alert_id}/resolve", json={"note": "handled"})
    assert res.status_code == 200
    assert res.json()["status"] == "resolved"

    _publish_entry(client, comp["entries"][2])
    ev3 = _evaluate(client, cid)
    hit3 = next(
        h for h in ev3.json()["hits"]
        if h["entry_id"] == comp["entries"][2]["entry_id"]
    )
    assert hit3["disposition"] == "escalated"
    assert hit3["reason"] == "new_hit_after_resolve"
    detail = client.get(f"/v1/alerts/{alert_id}").json()
    assert detail["status"] == "open"
    kinds = [e["kind"] for e in detail["events"]]
    assert kinds == [
        "opened", "acknowledged", "escalated", "resolved", "escalated"
    ]


# --------------------------------------------------------------------------- #
# operator transitions: acknowledge / resolve
# --------------------------------------------------------------------------- #
def _open_alert(client, prefix="A"):
    _process_at(client, prefix, BASE_DT)
    comp = _published_compilation(client, BASE_DT - DAY, BASE_DT + DAY)
    _make_rule(client, name=f"r-{prefix}")
    ev = _evaluate(client, comp["compilation_id"])
    return ev.json()["hits"][0]["alert_id"]


def test_acknowledge_and_resolve_idempotent(client):
    alert_id = _open_alert(client)

    ack = client.post(f"/v1/alerts/{alert_id}/acknowledge", json={})
    assert ack.status_code == 200, ack.text
    assert ack.json()["status"] == "acknowledged"
    again = client.post(
        f"/v1/alerts/{alert_id}/acknowledge", json={"note": "dup"}
    )
    assert again.status_code == 200
    detail = client.get(f"/v1/alerts/{alert_id}").json()
    kinds = [e["kind"] for e in detail["events"]]
    assert kinds == ["opened", "acknowledged"]  # no duplicate transition

    res = client.post(f"/v1/alerts/{alert_id}/resolve", json={"note": "done"})
    assert res.status_code == 200
    assert res.json()["status"] == "resolved"
    again = client.post(f"/v1/alerts/{alert_id}/resolve", json={})
    assert again.status_code == 200
    detail = client.get(f"/v1/alerts/{alert_id}").json()
    kinds = [e["kind"] for e in detail["events"]]
    assert kinds == ["opened", "acknowledged", "resolved"]
    # a resolved alert cannot be acknowledged
    late = client.post(f"/v1/alerts/{alert_id}/acknowledge", json={})
    assert late.status_code == 409
    assert late.json()["detail"]["code"] == "alert_already_resolved"

    missing = client.post("/v1/alerts/deadbeef/acknowledge", json={})
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "alert_not_found"


def test_overlapping_evaluations_share_hits_without_duplicates(client):
    comp = _three_quakes(client, prefix="O", gap_s=600.0)
    cid = comp["compilation_id"]
    r1 = _make_rule(client, name="ov-1", merge_interval_s=900.0)
    r2 = _make_rule(client, name="ov-2", merge_interval_s=900.0)

    first = _evaluate(client, cid, rule_ids=[r1["rule_id"]])
    assert first.status_code == 201
    assert first.json()["new_hit_count"] == 3

    # a second evaluation whose scope overlaps (r1) and extends (r2) the first
    second = _evaluate(client, cid, rule_ids=[r1["rule_id"], r2["rule_id"]])
    assert second.status_code == 201, second.text
    s = second.json()
    # (r1, entry) hits already exist: recorded only once, attributed to the
    # first evaluation; the second evaluation records only the r2 matches
    assert s["new_hit_count"] == 3
    assert s["hit_count"] == 3
    assert s["matched_count"] == 6  # full (r1|r2) x 3-entry scope is visible
    r1_hits = [h for h in s["hits"] if h["rule_id"] == r1["rule_id"]]
    assert {h["evaluation_id"] for h in r1_hits} == {
        first.json()["evaluation_id"]
    }
    # globally: exactly 6 hits and 2 alerts (one merged alert per rule)
    alerts = client.get("/v1/alerts").json()["alerts"]
    assert len(alerts) == 2
    assert sum(a["hit_count"] for a in alerts) == 6


def test_retired_rule_is_skipped_on_repass(client):
    starts = [BASE_DT + timedelta(seconds=600 * i) for i in range(2)]
    for i, start in enumerate(starts):
        _process_at(client, f"R{i}", start)
    comp = _compile_window(client, BASE_DT - DAY, BASE_DT + DAY)
    cid = comp["compilation_id"]
    rule = _make_rule(client, name="retiring", merge_interval_s=900.0)
    _publish_entry(client, comp["entries"][0])

    first = _evaluate(client, cid)
    assert first.status_code == 201
    assert first.json()["new_hit_count"] == 1

    # retire the rule, publish the second entry, re-evaluate: no new hits
    client.post(
        f"/v1/alert-rules/{rule['rule_id']}/retire",
        json={"expected_revision": 2},
    )
    _publish_entry(client, comp["entries"][1])
    second = _evaluate(client, cid)
    assert second.status_code == 200
    s = second.json()
    assert s["new_hit_count"] == 0
    assert s["rules"] == []  # retired rules no longer take part
    assert s["effective_rule_ids"] == []
    # the existing alert and its audit chain remain untouched
    alerts = client.get("/v1/alerts").json()["alerts"]
    assert len(alerts) == 1
    assert alerts[0]["status"] == "open"
    assert alerts[0]["hit_count"] == 1


# --------------------------------------------------------------------------- #
# catalog withdrawal: close only unacknowledged alerts, keep the audit chain
# --------------------------------------------------------------------------- #
def test_withdrawal_closes_only_open_alerts(client):
    starts = [BASE_DT + timedelta(seconds=3600 * i) for i in range(2)]
    for i, start in enumerate(starts):
        _process_at(client, f"W{i}", start)
    comp = _compile_window(client, BASE_DT - DAY, BASE_DT + DAY)
    cid = comp["compilation_id"]
    _make_rule(client, name="wd")
    for entry in comp["entries"]:
        _publish_entry(client, entry)
    ev = _evaluate(client, cid)
    # two far-apart events -> two independent open alerts
    alert_ids = [h["alert_id"] for h in ev.json()["hits"]]
    assert len(set(alert_ids)) == 2
    open_alert, ack_alert = sorted(alert_ids)

    # acknowledge one of them; the other stays open
    client.post(f"/v1/alerts/{ack_alert}/acknowledge", json={})

    # withdrawing the underlying entries closes only the open alert
    for entry in comp["entries"]:
        client.post(
            f"/v1/catalog/entries/{entry['entry_id']}/withdraw",
            json={"expected_revision": 2, "reason": "superseded"},
        )
    closed = client.get(f"/v1/alerts/{open_alert}").json()
    assert closed["status"] == "closed"
    kinds = [e["kind"] for e in closed["events"]]
    assert kinds == ["opened", "closed"]
    close_event = closed["events"][-1]
    assert close_event["reason"] == "catalog_entry_withdrawn"
    assert close_event["detail"]["withdraw_reason"] == "superseded"
    # the audit chain is preserved: hits and dispositions survive the close
    assert len(closed["hits"]) == 1
    assert closed["hits"][0]["disposition"] == "opened"

    kept = client.get(f"/v1/alerts/{ack_alert}").json()
    assert kept["status"] == "acknowledged"
    assert [e["kind"] for e in kept["events"]] == ["opened", "acknowledged"]

    # a closed alert rejects operator transitions
    for verb in ("acknowledge", "resolve"):
        resp = client.post(f"/v1/alerts/{open_alert}/{verb}", json={})
        assert resp.status_code == 409
        assert resp.json()["detail"]["code"] == "alert_closed"


def test_closed_alert_does_not_absorb_new_hits(client):
    # two quakes 600 s apart in one compilation
    _process_at(client, "Z0", BASE_DT)
    _process_at(client, "Z1", BASE_DT + timedelta(seconds=600))
    comp = _compile_window(client, BASE_DT - DAY, BASE_DT + DAY)
    cid = comp["compilation_id"]
    assert comp["entry_count"] == 2
    _make_rule(client, name="noclose", merge_interval_s=900.0)
    _publish_entry(client, comp["entries"][0])
    ev = _evaluate(client, cid)
    first_alert = ev.json()["hits"][0]["alert_id"]

    client.post(
        f"/v1/catalog/entries/{comp['entries'][0]['entry_id']}/withdraw",
        json={"expected_revision": 2, "reason": "false event"},
    )
    assert client.get(f"/v1/alerts/{first_alert}").json()["status"] == "closed"

    # a later entry within the merge interval opens a NEW alert instead of
    # merging into the closed one
    _publish_entry(client, comp["entries"][1])
    ev2 = _evaluate(client, cid)
    hit = next(
        h for h in ev2.json()["hits"]
        if h["entry_id"] == comp["entries"][1]["entry_id"]
    )
    assert hit["disposition"] == "opened"
    assert hit["alert_id"] != first_alert


# --------------------------------------------------------------------------- #
# concurrency: no duplicate or lost hits
# --------------------------------------------------------------------------- #
def test_concurrent_identical_evaluations_collapse(client):
    comp = _three_quakes(client, prefix="CC", gap_s=600.0)
    cid = comp["compilation_id"]
    _make_rule(client, name="conc", merge_interval_s=900.0)

    results = []
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        r = _evaluate(client, cid)
        results.append((r.status_code, r.json()["evaluation_id"]))

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start(); t2.start(); t1.join(); t2.join()

    statuses = sorted(s for s, _ in results)
    assert statuses == [200, 201], results
    assert results[0][1] == results[1][1]  # one surviving evaluation
    # exactly three hits recorded (no duplicates from the race)
    detail = client.get(f"/v1/alerts/evaluations/{results[0][1]}").json()
    assert detail["hit_count"] == 3
    assert detail["matched_count"] == 3
    assert len({h["hit_id"] for h in detail["hits"]}) == 3


def test_concurrent_evaluations_of_distinct_compilations(client):
    _process_at(client, "P0", BASE_DT)
    _process_at(client, "P1", BASE_DT + timedelta(seconds=600))
    _make_rule(client, name="par", merge_interval_s=900.0)
    comp1 = _published_compilation(
        client, BASE_DT - DAY, BASE_DT + timedelta(seconds=300)
    )
    comp2 = _published_compilation(
        client, BASE_DT + timedelta(seconds=300), BASE_DT + DAY
    )

    results = []
    barrier = threading.Barrier(2)

    def worker(cid):
        barrier.wait()
        r = _evaluate(client, cid)
        results.append(r.status_code)

    t1 = threading.Thread(target=worker, args=(comp1["compilation_id"],))
    t2 = threading.Thread(target=worker, args=(comp2["compilation_id"],))
    t1.start(); t2.start(); t1.join(); t2.join()

    assert sorted(results) == [201, 201]
    # both hits recorded exactly once, merged into a single alert
    alerts = client.get("/v1/alerts").json()["alerts"]
    assert len(alerts) == 1
    assert alerts[0]["hit_count"] == 2
    detail = client.get(f"/v1/alerts/{alerts[0]['alert_id']}").json()
    assert len(detail["hits"]) == 2
    assert {h["compilation_id"] for h in detail["hits"]} == {
        comp1["compilation_id"], comp2["compilation_id"]
    }


# --------------------------------------------------------------------------- #
# queries: filters and stable cursor pagination
# --------------------------------------------------------------------------- #
def test_alert_list_filters_and_pagination(client):
    starts = [BASE_DT + timedelta(seconds=3600 * i) for i in range(3)]
    for i, start in enumerate(starts):
        _process_at(client, f"L{i}", start)
    comp = _published_compilation(client, BASE_DT - DAY, BASE_DT + DAY)
    cid = comp["compilation_id"]
    rule = _make_rule(client, name="paged")
    ev = _evaluate(client, cid)
    assert ev.json()["new_hit_count"] == 3

    # keyset pagination: three pages of one, stable and non-overlapping
    seen, cursor = [], None
    for _ in range(6):
        params = {"limit": 1}
        if cursor:
            params["cursor"] = cursor
        page = client.get("/v1/alerts", params=params).json()
        seen.extend(a["alert_id"] for a in page["alerts"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert len(seen) == 3 and len(set(seen)) == 3

    # filters: rule / status / compilation
    by_rule = client.get("/v1/alerts", params={"rule_id": rule["rule_id"]}).json()
    assert len(by_rule["alerts"]) == 3
    open_only = client.get("/v1/alerts", params={"status": "open"}).json()
    assert len(open_only["alerts"]) == 3
    resolved = client.get("/v1/alerts", params={"status": "resolved"}).json()
    assert resolved["alerts"] == []
    by_comp = client.get("/v1/alerts", params={"compilation_id": cid}).json()
    assert len(by_comp["alerts"]) == 3
    other_comp = _published_compilation(
        client, BASE_DT + 2 * DAY, BASE_DT + 3 * DAY
    )
    empty = client.get(
        "/v1/alerts", params={"compilation_id": other_comp["compilation_id"]}
    ).json()
    assert empty["alerts"] == []

    # unknown filter targets are explicit 404s
    assert client.get(
        "/v1/alerts", params={"rule_id": "deadbeef"}
    ).status_code == 404
    assert client.get(
        "/v1/alerts", params={"compilation_id": "deadbeef"}
    ).status_code == 404

    # cursors are bound to their query fingerprint
    first_page = client.get("/v1/alerts", params={"limit": 1}).json()
    bad = client.get(
        "/v1/alerts",
        params={"limit": 1, "status": "open",
                "cursor": first_page["next_cursor"]},
    )
    assert bad.status_code == 422
    assert bad.json()["detail"]["code"] == "invalid_cursor"
    malformed = client.get("/v1/alerts", params={"cursor": "not-a-cursor"})
    assert malformed.status_code == 422
    assert malformed.json()["detail"]["code"] == "invalid_cursor"


def test_rule_and_evaluation_list_pagination(client):
    for i in range(3):
        _make_rule(client, name=f"pg-{i}")
    seen, cursor = [], None
    for _ in range(6):
        params = {"limit": 2}
        if cursor:
            params["cursor"] = cursor
        page = client.get("/v1/alert-rules", params=params).json()
        seen.extend(r["rule_id"] for r in page["rules"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert len(seen) == 3 and len(set(seen)) == 3

    _process_at(client, "V0", BASE_DT)
    comp = _published_compilation(client, BASE_DT - DAY, BASE_DT + DAY)
    _process_at(client, "V1", BASE_DT + 2 * DAY)
    comp2 = _published_compilation(client, BASE_DT + DAY, BASE_DT + 3 * DAY)
    for cid in (comp["compilation_id"], comp2["compilation_id"]):
        resp = _evaluate(client, cid)
        assert resp.status_code == 201

    page = client.get("/v1/alerts/evaluations", params={"limit": 1}).json()
    assert len(page["evaluations"]) == 1
    assert page["next_cursor"] is not None
    rest = client.get(
        "/v1/alerts/evaluations",
        params={"limit": 1, "cursor": page["next_cursor"]},
    ).json()
    assert len(rest["evaluations"]) == 1
    ids = {
        page["evaluations"][0]["evaluation_id"],
        rest["evaluations"][0]["evaluation_id"],
    }
    assert len(ids) == 2

    filtered = client.get(
        "/v1/alerts/evaluations",
        params={"compilation_id": comp["compilation_id"]},
    ).json()
    assert len(filtered["evaluations"]) == 1
    assert (
        filtered["evaluations"][0]["compilation_id"] == comp["compilation_id"]
    )
    missing = client.get(
        "/v1/alerts/evaluations", params={"compilation_id": "deadbeef"}
    )
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "catalog_compilation_not_found"

    detail = client.get(
        f"/v1/alerts/evaluations/{filtered['evaluations'][0]['evaluation_id']}"
    )
    assert detail.status_code == 200
    assert detail.json()["created"] is False
    gone = client.get("/v1/alerts/evaluations/deadbeef")
    assert gone.status_code == 404
    assert gone.json()["detail"]["code"] == "alert_evaluation_not_found"


def test_alert_detail_contains_hits_and_audit_chain(client):
    alert_id = _open_alert(client, prefix="D")
    detail = client.get(f"/v1/alerts/{alert_id}").json()
    assert detail["status"] == "open"
    assert detail["hit_count"] == 1
    assert detail["rule_name"] == "r-D"
    hit = detail["hits"][0]
    assert hit["disposition"] == "opened"
    assert set(hit["entry"]) == {
        "origin_time", "latitude", "longitude", "depth_km",
        "rms_residual_s", "station_count", "quality",
    }
    assert [e["seq"] for e in detail["events"]] == [1]
    assert detail["events"][0]["kind"] == "opened"
    assert detail["status_since"] is not None

    missing = client.get("/v1/alerts/deadbeef")
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "alert_not_found"


# --------------------------------------------------------------------------- #
# existing semantics remain intact
# --------------------------------------------------------------------------- #
def test_alerting_does_not_change_catalog_or_run_semantics(client):
    _process_at(client, "S0", BASE_DT)
    comp = _published_compilation(client, BASE_DT - DAY, BASE_DT + DAY)
    entry = comp["entries"][0]
    _make_rule(client, name="safe")
    ev = _evaluate(client, comp["compilation_id"])
    assert ev.status_code == 201

    # evaluation never touches catalog entries (revision/content unchanged)
    after = client.get(f"/v1/catalog/entries/{entry['entry_id']}").json()
    assert after["revision"] == 2  # only the publish bump
    assert after["status"] == "published"
    assert after["origin_time"] == entry["origin_time"]

    # the withdraw response schema is exactly the catalog entry payload
    resp = client.post(
        f"/v1/catalog/entries/{entry['entry_id']}/withdraw",
        json={"expected_revision": 2, "reason": "cleanup"},
    )
    assert resp.status_code == 200
    assert set(resp.json()) == {
        "entry_id", "compilation_id", "revision", "status", "origin_time",
        "latitude", "longitude", "depth_km", "rms_residual_s", "station_count",
        "quality", "master", "stations", "shared_stations",
        "distinct_run_count", "source_count", "sources", "created_at",
        "published_at", "withdrawn_at", "withdraw_reason",
    }
    assert resp.json()["status"] == "withdrawn"

    # run payloads are untouched as well
    run_id = after["master"]["run_id"]
    run = client.get(f"/v1/runs/{run_id}")
    assert run.status_code == 200
    assert run.json()["events"]
