import json
import re
import os
import random
import calendar
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    TISER_TRAIN_JSON,
    TISER_TEST_JSON,
    IMAGES_DIR,
    MM_TISER_TRAIN_JSON,
    MM_TISER_TEST_JSON,
)


def parse_date_decimal(date_str):
    """
    Converte una stringa di data in formato decimale.
    Esempi:
      '1917' -> (1917.0, False)
      'Jan, 1917' -> (1917.0, True)
    """
    if date_str is None:
        return None, False
    s = str(date_str).strip().replace(",", "")
    if re.fullmatch(r"\d{4}", s):
        return float(s), False
    try:
        parts = s.replace(",", "").split()
        if len(parts) == 2:
            month_str = parts[0]
            year_str = parts[-1]
            month_map = {abbr: i for i, abbr in enumerate(calendar.monthabbr) if abbr}
            if month_str not in month_map:
                month_map.update({name: i for i, name in enumerate(calendar.monthname) if name})
            m = month_map.get(month_str)
            if m is None:
                return float(year_str), False
            return int(year_str) + (m - 1) / 12.0, True
    except Exception:
        pass
    return None, False


def sanitize_filename(name: str) -> str:
    name = str(name)
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        name = name.replace(ch, "_")
    name = re.sub(r"\s+", "_", name)
    return name


def extract_temporal_data(prompt_text: str, dataset_name: str):
    """
    Estrae eventi temporali dal prompt.
    Gestisce TOT-Semantic con parser dedicato, e altri dataset con parser robusto.
    Ritorna (df, has_months).
    """
    if not isinstance(prompt_text, str):
        return None, False

    context = None
    markers = ["Temporal_context", "Temporal Context"]
    for marker in markers:
        if marker in prompt_text:
            context = prompt_text.split(marker, 1)[1].strip()
            for endmarker in ["Answer", "answer", "ANSWER", "Question"]:
                if endmarker in context:
                    context = context.split(endmarker, 1)[0].strip()
                    break
            break

    if not context:
        if len(prompt_text) < 2000:
            return None, False
        context = prompt_text

    events = []
    has_months = False
    ds = dataset_name.lower()

    # TOT-Semantic
    if "totsemantic" in ds:
        pattern_tot = re.compile(r"(E\d+)\s+was the\s+(R\d+)\s+of\s+(E\d+)\s+from\s+(\d{4})\s+to\s+(\d{4})")
        for match in pattern_tot.finditer(context):
            subj, rel, obj, ys, ye = match.groups()
            s = float(ys)
            e = float(ye)
            if e < s:
                s, e = e, s
            label = f"{subj} {rel} {obj}"
            events.append({"Task": label, "Start": s, "Finish": e, "Relation": rel})
        if events:
            df = pd.DataFrame(events)
            return df, False

    # pulizia generica
    clean_context = re.sub(r"\s+", " ", context)
    clean_context = re.sub(r"(\d)\.(\d)", r"\1DOT\2", clean_context)
    sentences = re.split(r"\.\s*", clean_context)

    # pattern 1: "YYYY - YYYY Description"
    for sent in sentences:
        sent = sent.replace("DOT", ".").strip()
        if not sent:
            continue
        match_range_colon = re.search(r"(\d{4})\s*-\s*(\d{4})\s*(.*)", sent)
        if match_range_colon:
            s = float(match_range_colon.group(1))
            e = float(match_range_colon.group(2))
            full_desc = match_range_colon.group(3).strip()
            leak = re.search(r"\d{4}\s*-\s*\d{4}", full_desc)
            if leak:
                full_desc = full_desc.split(leak.group(0), 1)[0]
            full_desc = full_desc.strip(" .,")

            inner = re.search(r"(.+)", full_desc)
            if inner:
                full_desc = inner.group(1).strip()
            events.append({"Task": full_desc, "Start": s, "Finish": e})
            continue

        # pattern 2: "Description from YYYY to YYYY" o con mesi
        match_from_to = re.search(r"(.+?)\s+from\s+([A-Za-z,\s\d]+)\s+to\s+([A-Za-z,\s\d]+)", sent)
        if match_from_to:
            full_desc = match_from_to.group(1).strip()
            starts = match_from_to.group(2).strip()
            ends = match_from_to.group(3).strip()
            s, hm1 = parse_date_decimal(starts)
            e, hm2 = parse_date_decimal(ends)
            if s is not None and e is not None:
                if hm1 or hm2:
                    has_months = True
                events.append({"Task": full_desc, "Start": s, "Finish": e})
            continue

    # fallback TGQA
    if not events and ("tgqa" in ds or "tgqatest" in ds):
        pattern_tgqa = re.compile(r"(.+?)\s+(starts|ends)\s+at\s+(\d{4})")
        matches = re.findall(pattern_tgqa, context)
        temp = {}
        for desc, t, year in matches:
            desc = desc.strip()
            d = temp.setdefault(desc, {"start": None, "end": None})
            v = float(year)
            if t == "starts":
                d["start"] = v
            else:
                d["end"] = v
        for desc, te in temp.items():
            s = te["start"]
            e = te["end"]
            if s is None and e is None:
                continue
            if s is None:
                s = e
            if e is None:
                e = s
            events.append({"Task": desc, "Start": s, "Finish": e})

    if not events:
        return None, False

    df = pd.DataFrame(events)
    df = df[df["Task"] != ""].dropna()
    df["Finish"] = df[["Start", "Finish"]].max(axis=1)
    df["Start"] = df[["Start", "Finish"]].min(axis=1)
    df = df.drop_duplicates()
    return df, has_months


def get_random_style():
    palettes = [
        plt.cm.Pastel1.colors,
        plt.cm.Set2.colors,
        plt.cm.tab10.colors,
        plt.cm.Set3.colors,
    ]
    backgrounds = ["#ffffff", "#f8f9fa", "#fffaf0", "#f0f8ff", "#f5f5f5"]
    fonts = ["sans-serif", "serif", "monospace"]
    return {
        "bg": random.choice(backgrounds),
        "colors": random.choice(palettes),
        "font": random.choice(fonts),
        "grid": True,
        "gridstyle": random.choice(["--", "-", "-."]),
        "titlesize": random.randint(12, 15),
    }


def setup_xaxis_smart(ax, df, has_months):
    all_vals = list(df["Start"]) + list(df["Finish"])
    minv, maxv = min(all_vals), max(all_vals)
    margin = max(0.5, (maxv - minv) * 0.05)
    ax.set_xlim(minv - margin, maxv + margin)

    if has_months:
        raw_ticks = sorted(set(all_vals))
    else:
        raw_ticks = sorted(list(set(int(round(x)) for x in all_vals)))

    if len(raw_ticks) > 1 and raw_ticks[-1] - raw_ticks[0] > 20:
        full_range = np.arange(raw_ticks[0], raw_ticks[-1] + 1, 1)
        if len(full_range) <= 15:
            raw_ticks = full_range

    if len(raw_ticks) > 15:
        step = int(np.ceil(len(raw_ticks) / 15))
        ticks = raw_ticks[::step]
    else:
        ticks = raw_ticks

    labels = []
    for t in ticks:
        if has_months:
            year = int(t)
            m = int(round((t - year) * 12)) + 1
            m = max(1, min(12, m))
            labels.append(f"{calendar.monthabbr[m]} {year}")
        else:
            labels.append(str(int(t)))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Time", fontsize=10, fontweight="bold")


def plot_totsemantic_gantt(ax, df, style):
    dfsorted = df.sort_values(["Start", "Finish"]).reset_index(drop=True)
    numrows = len(dfsorted)

    if "Relation" in dfsorted.columns:
        rels = sorted(dfsorted["Relation"].unique())
        cmap = plt.cm.get_cmap("tab20", len(rels))
        rel_to_color = {r: cmap(i) for i, r in enumerate(rels)}
    else:
        rels = []
        rel_to_color = {}

    for i, row in dfsorted.iterrows():
        width = max(0.2, row["Finish"] - row["Start"])
        ypos = i
        rel = row.get("Relation", None)
        c = rel_to_color.get(rel, style["colors"][i % len(style["colors"])])
        ax.barh(
            ypos,
            width,
            left=row["Start"],
            height=0.6,
            color=c,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        )
        label = str(row["Task"])
        if len(label) > 60:
            label = label[:57] + "..."
        ax.text(
            row["Start"],
            ypos + 0.35,
            label,
            fontsize=8,
            va="bottom",
            ha="left",
            fontfamily=style["font"],
            color="black",
            fontweight="medium",
        )

    ax.set_yticks([])
    ax.set_ylim(-1, numrows)
    ax.set_ylabel("Relations Structure", fontsize=10)
    ax.set_title("Relational Timeline Gantt", fontsize=style["titlesize"])
    if style["grid"]:
        ax.grid(True, axis="x", linestyle=style["gridstyle"], alpha=0.4)

    setup_xaxis_smart(ax, dfsorted, has_months=False)

    if rels:
        handles = [plt.Rectangle((0, 0), 1, 1, color=rel_to_color[r]) for r in rels]
        ax.legend(
            handles,
            rels,
            title="Relation Type",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=min(5, len(rels)),
            fontsize=8,
            frameon=False,
        )


def plot_generic_charts(ax, df, style, has_months, chart_type):
    df = df.sort_values("Start").reset_index(drop=True)
    colors = style["colors"]
    n = len(df)
    fs = 7 if n > 20 else 9

    if chart_type == "gantt":
        for i, row in df.iterrows():
            duration = max(0.1, row["Finish"] - row["Start"])
            c = colors[i % len(colors)]
            ax.barh(
                i,
                duration,
                left=row["Start"],
                height=0.5,
                color=c,
                edgecolor="black",
                alpha=0.85,
            )
            ax.text(
                row["Start"],
                i + 0.3,
                str(row["Task"]),
                fontsize=fs,
                ha="left",
                fontfamily=style["font"],
                fontweight="medium",
            )
        ax.set_title("Timeline Context Gantt", fontsize=style["titlesize"])

    elif chart_type == "scatter":
        for i, row in df.iterrows():
            c = colors[i % len(colors)]
            ax.hlines(
                i,
                df["Start"].min(),
                df["Finish"].max(),
                color="gray",
                alpha=0.1,
                linewidth=0.5,
                zorder=1,
            )
            ax.scatter(
                row["Start"],
                i,
                s=100,
                color=c,
                edgecolors="black",
                zorder=3,
            )
            if row["Finish"] - row["Start"] > 0.05:
                ax.scatter(
                    row["Finish"],
                    i,
                    s=100,
                    color=c,
                    marker="s",
                    edgecolors="black",
                    zorder=3,
                )
            ax.plot(
                [row["Start"], row["Finish"]],
                [i, i],
                color=c,
                linewidth=2,
                alpha=0.6,
                zorder=2,
            )
            ax.text(
                row["Start"],
                i + 0.2,
                str(row["Task"]),
                fontsize=fs,
                ha="left",
                fontfamily=style["font"],
            )
        ax.set_title("Event Intervals Scatter", fontsize=style["titlesize"])

    elif chart_type == "line":
        ax.hlines(
            range(n),
            df["Start"].min(),
            df["Finish"].max(),
            color="gray",
            alpha=0.2,
            linewidth=1,
        )
        for i, row in df.iterrows():
            y = i
            c = colors[i % len(colors)]
            ax.plot(
                [row["Start"], row["Finish"]],
                [y, y],
                color=c,
                linewidth=2,
                marker="o",
                markersize=6,
            )
            ax.annotate(
                str(row["Task"]),
                (row["Start"], y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="left",
                fontsize=fs,
                fontfamily=style["font"],
                bbox=dict(
                    boxstyle="round,pad=0.1",
                    fc="white",
                    ec="none",
                    alpha=0.6,
                ),
            )
        ax.set_title("Temporal Sequence", fontsize=style["titlesize"])

    ax.set_yticks([])
    ax.set_ylim(-1, n)
    if style["grid"]:
        ax.grid(True, axis="x", linestyle=style["gridstyle"], alpha=0.5)
    setup_xaxis_smart(ax, df, has_months)


def map_id(question_id: str) -> str:
    q = sanitize_filename(str(question_id))
    if q.startswith("story"):
        return q.split("Q")[0]
    parts = q.split("_")
    if len(parts) >= 2 and parts[-2] in ["easy", "hard", "L2", "L3"]:
        return "_".join(parts[:-2])
    return q


def worker_generate_chart(args):
    uid, prompt, dsname, outputdir = args
    df, has_months = extract_temporal_data(prompt, dsname)
    if df is None or df.empty:
        return None

    nev = len(df)
    safe = sanitize_filename(uid)
    imgpath = os.path.join(outputdir, f"{safe}.png")

    if os.path.exists(imgpath):
        return uid, imgpath, "gantt", nev

    style = get_random_style()
    figh = max(6, 0.8 * nev + 2)
    figw = 16

    try:
        fig, ax = plt.subplots(figsize=(figw, figh))
        fig.patch.set_facecolor(style["bg"])
        ax.set_facecolor(style["bg"])

        istot = "totsemantic" in dsname.lower()
        if istot:
            charttype = "gantt"
            plot_totsemantic_gantt(ax, df, style)
        else:
            charttype = random.choice(["gantt", "scatter", "line"])
            plot_generic_charts(ax, df, style, has_months, charttype)

        plt.savefig(imgpath, dpi=100, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        return uid, imgpath, charttype, nev
    except Exception:
        plt.close("all")
        return None


def generate_dataset_parallel(input_file, output_dir, json_output_file, num_workers=4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_examples = []
    unique_tasks = {}

    with open(input_file, "r", encoding="utf-8") as fin:
        lines = fin.readlines()
    for line in lines:
        try:
            ex = json.loads(line)
        except Exception:
            continue
        qid = ex.get("question_id") or ex.get("questionid") or ex.get("id")
        dsname = ex.get("datasetname") or ex.get("dataset_name")
        prompt = ex.get("prompt") or ex.get("input") or ex.get("text")
        if qid is None or prompt is None:
            continue
        uid = map_id(qid)
        ex["uid"] = uid
        all_examples.append(ex)
        if uid not in unique_tasks:
            unique_tasks[uid] = (prompt, dsname)

    print(f"Immagini uniche da generare {len(unique_tasks)}")

    tasks_args = [(uid, p, ds, output_dir) for uid, (p, ds) in unique_tasks.items()]
    uid_to_result = {}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(worker_generate_chart, arg): arg[0] for arg in tasks_args
        }
        for future in tqdm(as_completed(futures), total=len(tasks_args), desc="Generating Charts"):
            uid = futures[future]
            try:
                res = future.result()
                if res:
                    _, path, ctype, nev = res
                    uid_to_result[uid] = {"path": path, "type": ctype, "nev": nev}
            except Exception:
                continue

    print("Scrittura dataset finale...")
    success_count = 0
    with open(json_output_file, "w", encoding="utf-8") as fout:
        for q in tqdm(all_examples, desc="Merging JSON"):
            uid = q.pop("uid", None)
            if uid in uid_to_result:
                res = uid_to_result[uid]
                newentry = {
                    "id": q.get("question_id") or q.get("questionid") or q.get("id"),
                    "datasetname": q.get("datasetname") or q.get("dataset_name"),
                    "image": res["path"],
                    "question": q.get("question"),
                    "answer": q.get("answer"),
                    "output": q.get("output"),
                    "charttype": res["type"],
                    "numevents": int(res["nev"]),
                }
                fout.write(json.dumps(newentry, ensure_ascii=False) + "\n")
                success_count += 1

    print(f"COMPLETATO {success_count} righe scritte.")


if __name__ == "__main__":
    if os.path.exists(TISER_TRAIN_JSON):
        generate_dataset_parallel(TISER_TRAIN_JSON, IMAGES_DIR, MM_TISER_TRAIN_JSON, num_workers=os.cpu_count())
    if os.path.exists(TISER_TEST_JSON):
        generate_dataset_parallel(TISER_TEST_JSON, IMAGES_DIR, MM_TISER_TEST_JSON, num_workers=os.cpu_count())
