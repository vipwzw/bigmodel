"""生成《byte-level BPE 详解》的插图。

运行（需要 matplotlib，仓库 eval 依赖组已装）：
    MPLCONFIGDIR=/tmp/mplcfg .venv/bin/python docs/deep-dives/figures/make_figures.py

图号按文档中出现顺序编号（图 1–6），输出同目录 PNG（dpi=200）。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
font_manager.fontManager.addfont(FONT_PATH)
FONT = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams.update(
    {
        "font.family": FONT,
        "axes.unicode_minus": False,
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
    }
)

# 配色
BLUE, BLUE_L = "#4C72B0", "#DCE6F2"
ORANGE, ORANGE_L = "#DD8452", "#FBE5D6"
GREEN, GREEN_L = "#55A868", "#E2EFDA"
RED, RED_L = "#C44E52", "#FADBD8"
PURPLE, PURPLE_L = "#8172B3", "#EAE4F3"
BROWN, BROWN_L = "#937860", "#EDE4DB"
GRAY, GRAY_L = "#6B7280", "#F1F2F4"
INK = "#1F2937"


def box(ax, x, y, w, h, fc, ec, lw=1.2, r=0.018, z=2, ls="-"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z, linestyle=ls,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=1.5, rad=0.0, style="-|>", ms=13, ls="-", z=3):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style, mutation_scale=ms, color=color, linewidth=lw,
            connectionstyle=f"arc3,rad={rad}", linestyle=ls, zorder=z,
        )
    )


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def out(fig, name):
    fig.savefig(f"docs/deep-dives/figures/{name}.png", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ 图 1：流水线
def fig01_pipeline():
    fig, ax = canvas(15, 6.2)
    ax.text(0.5, 0.985, "byte-level BPE：文本 → 字节 → 字节符号 → 预分词 → token", ha="center", va="top",
            fontsize=16, color=INK)

    steps = [
        ("① 原始文本", "你好 world", "8 个 Unicode 字符", BLUE_L, BLUE),
        ("② UTF-8 字节", "E4 BD A0 E5 A5 BD 20\n77 6F 72 6C 64", "UTF-8 编码 · 12 个字节", ORANGE_L, ORANGE),
        ("③ 字节符号", "ä ½ ł å ¥ ½ Ġ\nw o r l d", "bytes_to_unicode · 每字节 1 符号", GREEN_L, GREEN),
        ("④ 预分词词块", "ä½łå¥½  |  Ġworld", "正则预分词 · 合并不跨块", PURPLE_L, PURPLE),
        ("⑤ Token（+ID）", "ä½ł  å¥½  Ġw  orld\n1730 996 7157 8161", "按 merge rank 贪心合并", BROWN_L, BROWN),
    ]
    w, gap, x0, y, h = 0.176, 0.030, 0.010, 0.46, 0.38
    centers = []
    for i, (title, body, note, fc, ec) in enumerate(steps):
        x = x0 + i * (w + gap)
        box(ax, x, y, w, h, fc, ec, lw=1.4)
        ax.text(x + w / 2, y + h - 0.055, title, ha="center", va="center", fontsize=12, color=INK)
        ax.text(x + w / 2, y + h * 0.50, body, ha="center", va="center", fontsize=12.5 if i != 1 else 10.8,
                color=INK, linespacing=1.6, family="monospace" if i in (1, 2) else FONT)
        ax.text(x + w / 2, y + 0.040, note, ha="center", va="center", fontsize=9.2, color=GRAY)
        centers.append((x, x + w))

    for i in range(len(steps) - 1):
        arrow(ax, centers[i][1] + 0.002, y + h / 2, centers[i + 1][0] - 0.002, y + h / 2, color=GRAY, lw=1.7, ms=15)

    box(ax, 0.03, 0.205, 0.94, 0.135, "#FFF8E7", "#D8B45A", lw=1.0)
    ax.text(0.05, 0.238, "训练阶段", ha="left", va="center", fontsize=10.5, color="#8A6D1D")
    ax.text(0.155, 0.238,
            "在一批语料上反复执行 ③→④→⑤：每轮统计最高频相邻对并记录一条 merge 规则；词表 = 特殊 token + 字节符号 + 合并出的新 token",
            ha="left", va="center", fontsize=10, color=INK)
    box(ax, 0.03, 0.055, 0.94, 0.135, "#EEF5EC", GREEN, lw=1.0)
    ax.text(0.05, 0.088, "推理阶段", ha="left", va="center", fontsize=10.5, color="#2F6B3C")
    ax.text(0.155, 0.088,
            "merge 列表冻结，只执行 ③→④→⑤；解码是逆向：token → 字节串 → UTF-8 解码（单独解一个字节碎片只会得到 U+FFFD）",
            ha="left", va="center", fontsize=10, color=INK)
    ax.text(0.5, 0.012,
            "中文 1 字 = 3 字节、emoji = 4 字节；空格 0x20 显示为 Ġ，换行 0x0A 显示为 Ċ（仅显示用，语义仍是字节）",
            ha="center", va="bottom", fontsize=9, color=GRAY)
    out(fig, "fig01_pipeline")


# ------------------------------------------------------------------ 图 2：训练循环
def fig03_train_loop():
    fig, ax = canvas(14.5, 6.6)
    ax.text(0.5, 0.975, "训练一轮 BPE：数对 → 选最高频 → 合并（四词玩具语料）", ha="center", va="top",
            fontsize=15.5, color=INK)

    # 左：语料
    box(ax, 0.015, 0.27, 0.30, 0.55, GRAY_L, "#B9BEC7", lw=1.2)
    ax.text(0.033, 0.785, "① 语料：预分词后按字节符号展开", fontsize=10.6, color=INK, va="center")
    rows = [("Ġ  l  o  w", "×5"), ("Ġ  l  o  w  e  r", "×2"), ("Ġ  n  e  w  e  s  t", "×6"),
            ("Ġ  w  i  d  e  s  t", "×3")]
    for i, (word, freq) in enumerate(rows):
        yy = 0.69 - i * 0.095
        ax.text(0.050, yy, word, fontsize=12, color=INK, va="center")
        ax.text(0.295, yy, freq, fontsize=11, color=GRAY, va="center", ha="right")
    ax.text(0.033, 0.315, "词频等价于把同一词重复计算；\n这样不必真的展开语料。", fontsize=8.8, color=GRAY, va="center")
    arrow(ax, 0.320, 0.545, 0.370, 0.545)
    ax.text(0.345, 0.585, "统计\n加权", fontsize=8.4, color=GRAY, ha="center")

    # 中：相邻对频次
    box(ax, 0.376, 0.27, 0.305, 0.55, "#FFF8E7", "#D8B45A", lw=1.2)
    ax.text(0.394, 0.785, "② 所有相邻对的加权频次（取最高）", fontsize=10.6, color=INK, va="center")
    pairs = [("e + s", 9, True), ("s + t", 9, True), ("w + e", 8, False), ("Ġ + l", 7, False),
             ("l + o", 7, False), ("o + w", 7, False), ("…", None, False)]
    for i, (pair, cnt, hot) in enumerate(pairs):
        yy = 0.705 - i * 0.058
        ax.text(0.396, yy, pair, fontsize=10.5, color=RED if hot else INK, va="center")
        if cnt is not None:
            bar_x = 0.560
            ax.add_patch(Rectangle((bar_x, yy - 0.013), 0.0075 * cnt, 0.026,
                                   facecolor=RED if hot else BLUE, alpha=0.85, lw=0, zorder=2))
            ax.text(bar_x + 0.0075 * cnt + 0.006, yy, str(cnt), fontsize=9.5, color=INK, va="center")
    ax.text(0.394, 0.310, "平票（e+s 与 s+t 都是 9）：按 (id_a, id_b) 最小取 e(5)+s(11)。\n"
                          "这是实现细节；本机 tokenizers 0.23.3 实测如此。", fontsize=8.5, color=GRAY, va="center")
    arrow(ax, 0.683, 0.545, 0.733, 0.545)
    ax.text(0.708, 0.585, "选出\n一对", fontsize=8.4, color=GRAY, ha="center")

    # 右：合并结果
    box(ax, 0.739, 0.27, 0.246, 0.55, BLUE_L, BLUE, lw=1.2)
    ax.text(0.757, 0.785, "③ 合并 e+s → es（新 id = 15）", fontsize=10.6, color=INK, va="center")
    merged = [("Ġ  l  o  w", "×5"), ("Ġ  l  o  w  e  r", "×2"), ("Ġ  n  e  w  es  t", "×6"),
              ("Ġ  w  i  d  es  t", "×3")]
    for i, (word, freq) in enumerate(merged):
        yy = 0.69 - i * 0.095
        ax.text(0.755, yy, word, fontsize=12, color=INK, va="center")
        ax.text(0.970, yy, freq, fontsize=11, color=GRAY, va="center", ha="right")
    ax.text(0.755, 0.360, "每处 e+s 少一个符号：", fontsize=8.8, color=GRAY, va="center")
    ax.text(0.755, 0.315, "语料符号数 95 → 86", fontsize=10, color=RED, va="center")

    arrow(ax, 0.86, 0.262, 0.17, 0.262, color=PURPLE, lw=1.6, rad=-0.10, style="-|>", ms=14)
    ax.text(0.515, 0.075, "重复：数对 → 选最高频 → 合并（每轮词表 +1），直到词表达到目标大小，或最高频 < min_frequency",
            ha="center", fontsize=9.6, color=PURPLE)
    ax.text(0.5, 0.022, "注：真实训练的初始词表是 256 个字节符号 + 特殊 token；玩具语料只出现了 11 个符号，所以从 11 开始。",
            ha="center", fontsize=9, color=GRAY)
    out(fig, "fig03_train_loop")


# ------------------------------------------------------------------ 图 3：合并树
def fig04_merge_tree():
    fig, ax = canvas(12.5, 7.2)
    ax.text(0.5, 0.985, "一个 token 的诞生：Ġnewest（id 23）的合并树", ha="center", va="top",
            fontsize=15.5, color=INK)
    ax.text(0.5, 0.925, "叶子 = 字节符号；内部节点 = merges 里的一条规则（merge #n 为合并轮次，t 为 token id）",
            ha="center", va="top", fontsize=10, color=GRAY)

    leaves = {"Ġ": (0.095, 0.10, "Ġ", 14), "n": (0.335, 0.10, "n", 8), "e1": (0.435, 0.10, "e", 5),
              "w": (0.545, 0.10, "w", 13), "e2": (0.655, 0.10, "e", 5), "s": (0.775, 0.10, "s", 11),
              "t": (0.885, 0.10, "t", 12)}
    internals = {
        "Ġnewest": (0.50, 0.86, "Ġnewest", 23, 8, PURPLE, PURPLE_L),
        "Ġnew": (0.28, 0.68, "Ġnew", 22, 7, BLUE, BLUE_L),
        "est": (0.76, 0.68, "est", 16, 1, GREEN, GREEN_L),
        "new": (0.40, 0.50, "new", 21, 6, ORANGE, ORANGE_L),
        "ew": (0.50, 0.32, "ew", 20, 5, BROWN, BROWN_L),
        "es": (0.715, 0.32, "es", 15, 0, RED, RED_L),
    }
    edges = [
        ("Ġnewest", "Ġnew"), ("Ġnewest", "est"), ("Ġnew", "Ġn"), ("Ġnew", "new"),
        ("new", "n"), ("new", "ew"), ("ew", "e1"), ("ew", "w"),
        ("est", "es"), ("est", "t"), ("es", "e2"), ("es", "s"),
    ]
    leaf_alias = {"Ġn": "Ġ"}

    def pos(name):
        if name in internals:
            return internals[name][0], internals[name][1]
        if name in leaves:
            return leaves[name][0], leaves[name][1]
        base = leaf_alias[name]
        return leaves[base][0], leaves[base][1]

    for parent, child in edges:
        x1, y1 = pos(parent)
        x2, y2 = pos(child)
        ax.plot([x1, x2], [y1 - 0.045, y2 + 0.045], color="#9AA1AC", lw=1.4, zorder=1, solid_capstyle="round")

    for key, (x, y, label, tid) in leaves.items():
        box(ax, x - 0.036, y - 0.038, 0.072, 0.076, GREEN_L, GREEN, lw=1.2, r=0.012)
        ax.text(x, y + 0.004, label, ha="center", va="center", fontsize=15, color=INK)
        ax.text(x, y - 0.058, f"t{tid}", ha="center", va="center", fontsize=8, color=GRAY)

    for key, (x, y, label, tid, rank, ec, fc) in internals.items():
        box(ax, x - 0.075, y - 0.043, 0.15, 0.086, fc, ec, lw=1.4, r=0.014)
        ax.text(x, y + 0.008, label, ha="center", va="center", fontsize=12.5, color=INK)
        ax.text(x, y - 0.027, f"t{tid} · merge #{rank}", ha="center", va="center", fontsize=7.8, color=ec)

    ax.text(0.045, 0.79, "解码：所有叶子拼起来 = 字节串\n20 6E 65 77 65 73 74 → “ newest”",
            fontsize=9.2, color=GRAY, va="center")
    ax.text(0.045, 0.42, "同一 token 可被不同词复用：\nes 也出现在 widest 里。", fontsize=9.2, color=GRAY, va="center")
    ax.text(0.5, 0.012, "注意：叶子是字节符号而不是字符——“Ġ”代表 0x20 空格；根节点 Ġnewest 才是完整字符的字节串。",
            ha="center", fontsize=9, color=GRAY)
    out(fig, "fig04_merge_tree")


# ------------------------------------------------------------------ 图 4：编码贪心
def fig05_encoding():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6), gridspec_kw={"wspace": 0.16})
    fig.suptitle("编码 = 反复合并 rank 最小的相邻对（rank = 它在 merges 列表中的下标）", fontsize=15.5,
                 color=INK, y=0.97)
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    def draw(ax, steps, title, subtitle):
        ax.text(0.5, 0.93, title, ha="center", va="center", fontsize=13, color=INK)
        ax.text(0.5, 0.875, subtitle, ha="center", va="center", fontsize=9.6, color=GRAY)
        n = len(steps)
        top, bottom = 0.78, 0.14
        ys = [top - i * (top - bottom) / max(n - 1, 1) for i in range(n)]
        for i, (syms, merged_idx, rank) in enumerate(steps):
            y = ys[i]
            unit, gap = 0.078, 0.012
            widths = [max(len(s) * 0.62, 1) for s in syms]
            total = sum(widths) * unit + (len(syms) - 1) * gap
            x = 0.5 - total / 2
            for j, (sym, wide) in enumerate(zip(syms, widths)):
                color = (RED, RED_L) if (merged_idx is not None and j == merged_idx) else (GRAY, "white")
                box(ax, x, y - 0.032, wide * unit, 0.064, color[1], color[0], lw=1.3, r=0.012)
                ax.text(x + wide * unit / 2, y, sym, ha="center", va="center", fontsize=13, color=INK)
                x += wide * unit + gap
            if rank is not None:
                ax.text(0.985, y, f"rank {rank}", ha="right", va="center", fontsize=9, color=RED)
            if i == 0:
                ax.text(0.02, y, "初始", ha="left", va="center", fontsize=9, color=GRAY)
            if i == n - 1:
                ax.text(0.02, y, "结果", ha="left", va="center", fontsize=9, color=GRAY)

    draw(
        axes[0],
        [
            (["l", "o", "w", "e", "s", "t"], None, None),
            (["l", "o", "w", "es", "t"], 3, 0),
            (["l", "o", "w", "est"], 3, 1),
            (["lo", "w", "est"], 0, 2),
        ],
        "lowest（无前导空格）",
        "3 次合并 → [lo] [w] [est]",
    )
    draw(
        axes[1],
        [
            (["Ġ", "l", "o", "w", "e", "s", "t"], None, None),
            (["Ġ", "l", "o", "w", "es", "t"], 4, 0),
            (["Ġ", "l", "o", "w", "est"], 4, 1),
            (["Ġ", "lo", "w", "est"], 1, 2),
            (["Ġlo", "w", "est"], 0, 3),
            (["Ġlow", "est"], 0, 4),
        ],
        " lowest（带前导空格，Ġ = 0x20）",
        "5 次合并 → [Ġlow] [est]：词表里恰好有 Ġlow，于是多吃掉一个空格",
    )
    fig.text(0.5, 0.015,
             "每轮只做一件事：找 rank 最小的相邻对并合并（可同时合并该对的所有不重叠出现）；红色 = 本轮合并出的单元。"
             "若某对不在 merges 里，就永远不合并。",
             ha="center", va="bottom", fontsize=9.5, color=GRAY)
    out(fig, "fig05_encoding")


# ------------------------------------------------------------------ 图 5：真实分词器对比
TOKENS_REPO = [
    ("ä½ł", "你"), ("å¥½", "好"), ("ï", None), ("¼", None), ("Į", None),
    ("ä¸ĸçķĮ", "世界"), ("ï", None), ("¼", None), ("ģ", None),
    ("H", "H"), ("ell", "ell"), ("o", "o"), ("ĠWorld", " World"), ("Ġ", " "),
    ("ð", None), ("Ł", None), ("ĺ", None), ("Ģ", None),
]
TOKENS_QWEN = [
    ("ä½łå¥½", "你好"), ("ï¼Į", "，"), ("ä¸ĸçķĮ", "世界"), ("ï¼ģ", "！"),
    ("Hello", "Hello"), ("ĠWorld", " World"), ("ĠðŁĺĢ", "<emoji>"),
]


def fig06_real():
    fig, ax = canvas(15, 7.6)
    ax.text(0.5, 0.985, "同一句话，两套 byte-level BPE 词表：16,384 vs 151,643", ha="center", va="top",
            fontsize=15.5, color=INK)
    ax.text(0.5, 0.928, "句子：你好，世界！Hello World [emoji U+1F600] —— 共 34 字节（中文 18 + ASCII 11 + 空格 1 + emoji 4）",
            ha="center", va="top", fontsize=10.5, color=GRAY)

    def draw_row(tokens, y, label, palette):
        ax.text(0.012, y + 0.135, label, ha="left", va="center", fontsize=10.2, color=INK)
        ax.text(0.988, y + 0.135, f"{len(tokens)} token", ha="right", va="center", fontsize=10, color=RED)
        total = sum(len(t[0]) for t in tokens)
        scale = 0.94 / total
        x = 0.012
        for i, (raw, dec) in enumerate(tokens):
            w = len(raw) * scale
            if dec is None:
                ec, fc, label_txt = RED, RED_L, raw
            else:
                ec, fc = palette[i % 2]
                label_txt = dec
            box(ax, x, y, w, 0.10, fc, ec, lw=1.0, r=0.006)
            fs = 7.4 if len(label_txt) > 3 else 8.4
            ax.text(x + w / 2, y + 0.05, label_txt, ha="center", va="center", fontsize=fs, color=INK)
            x += w

    draw_row(TOKENS_REPO, 0.60, "本仓库 bpe_16384（199 个字节符号 + 16,181 条 merge）",
             [(BLUE, BLUE_L), (ORANGE, ORANGE_L)])
    draw_row(TOKENS_QWEN, 0.33, "Qwen3-0.6B（256 个字节符号 + 151,387 条 merge + 26 个 added token）",
             [(GREEN, GREEN_L), (PURPLE, PURPLE_L)])

    ax.text(0.012, 0.565, "红色 = 单独解码不是合法 UTF-8 的字节碎片（标签显示原始字节符号）；拼起来才还原成完整字符。",
            fontsize=9.2, color=GRAY, va="center")
    ax.text(0.012, 0.275, "本仓库 16k 词表没学过全角标点“，”和“！”（训练语料清洗成半角），于是它们各被拆成 3 个字节碎片。",
            fontsize=9.2, color=GRAY, va="center")

    box(ax, 0.05, 0.09, 0.42, 0.145, "#FFF8E7", "#D8B45A", lw=1.0)
    ax.text(0.065, 0.195, "emoji U+1F600：本仓库 4 token（ð Ł ĺ Ģ）", fontsize=9.3, color=INK, va="center")
    ax.text(0.065, 0.150, "                        Qwen3 1 token（ĠðŁĺĢ = 空格+emoji）", fontsize=9.3, color=INK, va="center")
    ax.text(0.065, 0.115, "差别来自训练语料里 emoji 的出现频率", fontsize=8.4, color=GRAY, va="center")
    box(ax, 0.53, 0.09, 0.42, 0.145, "#EEF5EC", GREEN, lw=1.0)
    ax.text(0.545, 0.195, "制表符 \\t：本仓库未修复时 → <unk>（字节 0x09 不在 199 个符号里）", fontsize=8.9, color=INK, va="center")
    ax.text(0.545, 0.150, "                  加 initial_alphabet 后 → ĉ（1 token，无 UNK）", fontsize=8.9, color=INK, va="center")
    ax.text(0.545, 0.115, "Qwen3 词表含全部 256 个字节符号，不会有这种 UNK", fontsize=8.4, color=GRAY, va="center")
    out(fig, "fig06_real_tokenizers")


# ------------------------------------------------------------------ 图 6：UTF-8 与字节映射
def fig02_bytes():
    fig, ax = canvas(14, 5.9)
    ax.text(0.5, 0.985, "UTF-8 字节 → bytes_to_unicode 显示符号（256 个字节的双射）", ha="center", va="top",
            fontsize=15.5, color=INK)
    ax.text(0.5, 0.925, "可打印字节原样显示；控制字节/空白/软连字符等 68 个字节映射到 U+0100 起的可见字符。"
                        "映射只为显示与词表读写，语义永远是字节。",
            ha="center", va="top", fontsize=9.8, color=GRAY)

    rows = [
        ("你", "U+4F60", ["E4", "BD", "A0"], ["ä", "½", "ł"], "3 字节：1110xxxx + 2 个续字节"),
        ("好", "U+597D", ["E5", "A5", "BD"], ["å", "¥", "½"], "3 字节：1110xxxx + 2 个续字节"),
        ("emoji", "U+1F600", ["F0", "9F", "98", "80"], ["ð", "Ł", "ĺ", "Ģ"], "4 字节：11110xxx + 3 个续字节"),
        ("a", "U+0061", ["61"], ["a"], "ASCII：1 字节，自映射"),
        ("空格", "U+0020", ["20"], ["Ġ"], "空格 → Ġ（词表里看到 Ġ 就是空格）"),
        ("换行", "U+000A", ["0A"], ["Ċ"], "换行 → Ċ"),
    ]
    top, step = 0.81, 0.125
    for i, (ch, code, byte_hex, syms, note) in enumerate(rows):
        y = top - i * step
        ax.text(0.045, y, ch, ha="center", va="center", fontsize=13 if len(ch) > 1 else 15, color=INK)
        ax.text(0.045, y - 0.045, code, ha="center", va="center", fontsize=7.6, color=GRAY)
        x = 0.10
        for j, (hx, sym) in enumerate(zip(byte_hex, syms)):
            fc, ec = (BLUE_L, BLUE) if j == 0 else (ORANGE_L, ORANGE)
            box(ax, x, y - 0.032, 0.070, 0.064, fc, ec, lw=1.2, r=0.010)
            ax.text(x + 0.035, y, hx, ha="center", va="center", fontsize=10.5, family="monospace", color=INK)
            x += 0.078
        arrow(ax, x + 0.005, y, x + 0.080, y, color=GRAY, lw=1.2)
        ax.text(x + 0.043, y + 0.030, "映射", ha="center", va="center", fontsize=7.4, color=GRAY)
        x += 0.090
        for sym in syms:
            box(ax, x, y - 0.032, 0.070, 0.064, GREEN_L, GREEN, lw=1.2, r=0.010)
            ax.text(x + 0.035, y, sym, ha="center", va="center", fontsize=13, color=INK)
            x += 0.078
        ax.text(x + 0.02, y, note, ha="left", va="center", fontsize=9.2, color=GRAY)
    ax.text(0.045, 0.012, "图例：蓝 = UTF-8 首字节，橙 = 续字节（10xxxxxx）；绿 = 词表里的单字节符号（其 id 按 Unicode 码点排序）。",
            ha="left", va="bottom", fontsize=9, color=GRAY)
    out(fig, "fig02_bytes")


if __name__ == "__main__":
    fig01_pipeline()
    fig02_bytes()
    fig03_train_loop()
    fig04_merge_tree()
    fig05_encoding()
    fig06_real()
