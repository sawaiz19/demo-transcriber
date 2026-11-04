"""Cross-check two ASR models on the SAME diarized segments and highlight where they disagree.

Efficiency: diarization, audio decoding and segment boundaries are shared, so the second model
only costs one extra transcription pass over the existing chunks (models run one after the
other, so only one is ever in memory).
"""
import json

import streamlit as st

import compare
import pipeline
import postprocess


def align_segments(segs_a, segs_b, fold=None):
    """Per-segment word alignment of model A vs model B (same segment boundaries)."""
    rows = []
    for a, b in zip(segs_a, segs_b):
        ta, tb = a["text"].strip(), b["text"].strip()
        if not ta and not tb:
            continue
        aligned = compare.align(ta, tb, fold)
        stats = compare.summary(aligned)
        rows.append(
            {
                "index": a["index"],
                "start": a["start"],
                "end": a["end"],
                "speaker": a["speaker"],
                "is_enrolled": a.get("is_enrolled", False),
                "a": ta,
                "b": tb,
                "aligned": aligned,
                "agreement": stats["agreement"],
                "disagreements": stats["disagreements"],
            }
        )
    return rows


def _stamp(sec):
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def merged_segments(rows, segs_a, choices):
    """Final segments: chosen side for disagreeing segments, model A elsewhere."""
    by_index = {r["index"]: r for r in rows}
    out = []
    for seg in segs_a:
        r = by_index.get(seg["index"])
        text = seg["text"]
        if r and r["disagreements"]:
            text = r["b"] if choices.get(seg["index"], "A") == "B" else r["a"]
        out.append({**seg, "text": text})
    return out


def render(job, file_name, model_a, model_b, review_threshold=0.85):
    segs_a, segs_b = job["segments"], job["segments_b"]
    fold = postprocess.make_folder(postprocess.load_spelling())
    rows = align_segments(segs_a, segs_b, fold)
    if not rows:
        st.info("Nothing to compare yet.")
        return

    all_aligned = [s for r in rows for s in r["aligned"]]
    stats = compare.summary(all_aligned)
    differing = [r for r in rows if r["disagreements"]]
    review = [r for r in differing if r["agreement"] < review_threshold]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Word agreement", f"{stats['agreement'] * 100:.1f}%")
    c2.metric("Disagreeing spots", stats["disagreements"])
    c3.metric("Segments with differences", f"{len(differing)} / {len(rows)}")
    c4.metric(f"Need review (<{review_threshold * 100:.0f}%)", len(review))
    st.caption(
        f"**A:** {model_a.split(' (')[0]}  ·  **B:** {model_b.split(' (')[0]}. "
        "Roman spelling variants (e.g. nahi / nahin) are treated as equal, so only real differences are highlighted."
    )

    f1, f2, f3 = st.columns([2, 2, 3])
    prefer = f1.radio("Default choice when they differ", ["A", "B"], horizontal=True, key=f"xc_pref_{job['key'][:8]}")
    only_review = f2.checkbox("Only segments needing review", value=bool(review), key=f"xc_only_{job['key'][:8]}")
    worst_first = f3.checkbox("Lowest agreement first", value=True, key=f"xc_sort_{job['key'][:8]}")

    shown = review if only_review else differing
    if worst_first:
        shown = sorted(shown, key=lambda r: r["agreement"])

    choices = {}
    with st.container(height=560, border=True):
        if not shown:
            st.success("The two models agree on everything shown here.")
        for r in shown:
            sev = "🔴" if r["agreement"] < review_threshold else "🟡"
            st.markdown(
                f"{sev} **{r['speaker']}** · `{_stamp(r['start'])} – {_stamp(r['end'])}` · "
                f"agreement {r['agreement'] * 100:.0f}% · {r['disagreements']} difference(s)"
            )
            ca, cb = st.columns(2)
            with ca:
                st.markdown("**A**")
                st.markdown(compare.highlight_html(r["aligned"], "a"), unsafe_allow_html=True)
            with cb:
                st.markdown("**B**")
                st.markdown(compare.highlight_html(r["aligned"], "b"), unsafe_allow_html=True)
            choices[r["index"]] = st.radio(
                "Use", ["A", "B"], index=0 if prefer == "A" else 1, horizontal=True,
                key=f"xc_{prefer}_{job['key'][:8]}_{r['index']}", label_visibility="collapsed",
            )
            st.divider()

    # segments not on screen (filtered out) follow the default choice
    for r in differing:
        choices.setdefault(r["index"], prefer)

    final = merged_segments(rows, segs_a, choices)
    merged_text = pipeline.format_segments_as_dialogue(final, include_timestamps=True)
    base = file_name.rsplit(".", 1)[0]

    d1, d2 = st.columns(2)
    d1.download_button(
        "Download merged transcript (.txt)", merged_text + "\n",
        file_name=f"{base}_crosschecked.txt", mime="text/plain", use_container_width=True,
    )
    report = {
        "model_a": model_a, "model_b": model_b,
        "agreement": stats["agreement"], "disagreeing_spots": stats["disagreements"],
        "segments": [
            {k: r[k] for k in ("index", "start", "end", "speaker", "a", "b", "agreement", "disagreements")}
            | {"chosen": choices.get(r["index"], "A")}
            for r in differing
        ],
    }
    d2.download_button(
        "Download disagreement report (.json)", json.dumps(report, indent=2, ensure_ascii=False),
        file_name=f"{base}_disagreements.json", mime="application/json", use_container_width=True,
    )
    with st.expander("Preview merged transcript"):
        st.text(merged_text)
