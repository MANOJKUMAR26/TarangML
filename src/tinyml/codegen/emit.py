"""Turn a model artifact into one portable C99 file, a header and a C++ wrapper.

Codegen reads only model.json (docs/contracts/model-artifact.md). The C feature code follows
docs/contracts/features.md line by line, in float32, with sequential accumulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from tinyml import __version__
from tinyml.features.spec import NUM_BANDS, fft_size

TEMPLATES = Path(__file__).with_name("templates")


def cf(value: float) -> str:
    """Exact float32 literal."""
    v = np.float32(value)
    if not np.isfinite(v):
        raise ValueError(f"non-finite constant {value}")
    return np.format_float_scientific(v, unique=True, trim="-", exp_digits=2) + "f"


def int_type(values: list[int]) -> str:
    lo, hi = (min(values), max(values)) if values else (0, 0)
    for name, bits in (("int8_t", 8), ("int16_t", 16), ("int32_t", 32)):
        if -(1 << (bits - 1)) <= lo and hi < (1 << (bits - 1)):
            return name
    raise ValueError("value out of int32 range")


def c_array(ctype: str, name: str, values: list[Any], fmt=str, per_line: int = 8, const: bool = True) -> str:
    items = [fmt(v) for v in values] or ["0"]
    lines = []
    for i in range(0, len(items), per_line):
        lines.append("    " + ", ".join(items[i:i + per_line]) + ",")
    qualifier = "static const " if const else "static "
    return f"{qualifier}{ctype} {name}[{max(len(values), 1)}] = {{\n" + "\n".join(lines) + "\n};\n"


def c_string(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


@dataclass
class Generated:
    header: str
    source: str
    cpp: str
    example: str


class FeatureEmitter:
    """C for the feature list, computing only what the model uses."""

    def __init__(self, artifact: dict[str, Any]):
        self.window = artifact["window"]
        self.features = artifact["features"]
        self.n = int(self.window["samples"])
        self.axes = self.window["axes"]
        self.rate = float(self.window["sample_rate_hz"])
        self.n_fft = fft_size(self.n)
        self.spectral = any(f["kind"] in ("domfreq", "centroid", "band0", "band1", "band2", "band3")
                            for f in self.features)

    def channels(self) -> list[tuple[int, list[tuple[int, str]]]]:
        order: dict[int, list[tuple[int, str]]] = {}
        for pos, f in enumerate(self.features):
            order.setdefault(int(f["channel_index"]), []).append((pos, f["kind"]))
        return list(order.items())

    def sample_expr(self, channel: int) -> str:
        if channel >= 0:
            return f"window[i * TINYML_NUM_AXES + {channel}]"
        return "tm_mag_buf[i]"

    def emit(self) -> str:
        out: list[str] = []
        uses_mag = any(ch < 0 for ch, _ in self.channels())
        if uses_mag:
            terms = " + ".join(f"window[base + {a}] * window[base + {a}]" for a in range(len(self.axes)))
            out.append(
                f"static float tm_mag_buf[{self.n}];\n\n"
                "/* Vector magnitude of all axes, computed once per window (one square root per sample). */\n"
                "static void tm_fill_mag(const float *window)\n{\n"
                f"    for (int i = 0; i < {self.n}; i++) {{\n"
                "        const int base = i * TINYML_NUM_AXES;\n"
                f"        tm_mag_buf[i] = tm_sqrtf({terms});\n    }}\n}}\n"
            )
        if self.spectral:
            out.append(self.emit_fft())
        for channel, items in self.channels():
            out.append(self.emit_channel(channel, items))
        calls = "\n".join(
            f"    tm_channel_{self.cname(ch)}({'window, ' if ch >= 0 else ''}features);" for ch, _ in self.channels()
        )
        if uses_mag:
            calls = "    tm_fill_mag(window);\n" + calls
        out.append(f"void model_features(const float *window, float *features)\n{{\n{calls}\n}}\n")
        return "\n".join(out)

    def cname(self, channel: int) -> str:
        return "mag" if channel < 0 else str(channel)

    def emit_fft(self) -> str:
        half = self.n_fft // 2
        cos = [math.cos(2 * math.pi * k / self.n_fft) for k in range(half)]
        sin = [math.sin(2 * math.pi * k / self.n_fft) for k in range(half)]
        return (
            f"#define TM_FFT_SIZE {self.n_fft}\n"
            + c_array("float", "tm_twiddle_cos", cos, cf, 6)
            + c_array("float", "tm_twiddle_sin", sin, cf, 6)
            + "static float tm_fft_re[TM_FFT_SIZE];\nstatic float tm_fft_im[TM_FFT_SIZE];\n\n"
            "static void tm_fft(float *re, float *im)\n{\n"
            "    int i, j = 0;\n"
            "    for (i = 1; i < TM_FFT_SIZE; i++) {\n"
            "        int bit = TM_FFT_SIZE >> 1;\n"
            "        for (; j & bit; bit >>= 1) {\n            j ^= bit;\n        }\n"
            "        j ^= bit;\n"
            "        if (i < j) {\n"
            "            float t = re[i]; re[i] = re[j]; re[j] = t;\n"
            "            t = im[i]; im[i] = im[j]; im[j] = t;\n"
            "        }\n    }\n"
            "    for (int len = 2; len <= TM_FFT_SIZE; len <<= 1) {\n"
            "        const int half = len >> 1;\n"
            "        const int step = TM_FFT_SIZE / len;\n"
            "        for (i = 0; i < TM_FFT_SIZE; i += len) {\n"
            "            for (int k = 0; k < half; k++) {\n"
            "                const float wr = tm_twiddle_cos[k * step];\n"
            "                const float wi = -tm_twiddle_sin[k * step];\n"
            "                const int a = i + k, b = i + k + half;\n"
            "                const float tr = re[b] * wr - im[b] * wi;\n"
            "                const float ti = re[b] * wi + im[b] * wr;\n"
            "                re[b] = re[a] - tr;\n                im[b] = im[a] - ti;\n"
            "                re[a] += tr;\n                im[a] += ti;\n"
            "            }\n        }\n    }\n}\n"
        )

    def emit_channel(self, channel: int, items: list[tuple[int, str]]) -> str:
        kinds = {k for _, k in items}
        spectral = bool(kinds & {"domfreq", "centroid", "band0", "band1", "band2", "band3"})
        need_sq = bool(kinds & {"rms", "crest"})
        need_peak = bool(kinds & {"peak", "crest", "kurt"})
        need_minmax = "p2p" in kinds
        need_m2 = bool(kinds & {"std", "kurt"})
        need_m4 = "kurt" in kinds
        need_zcr = "zcr" in kinds
        need_mean = bool(kinds & {"mean"}) or need_m2 or need_zcr or spectral
        x = self.sample_expr(channel)
        params = "const float *window, float *features" if channel >= 0 else "float *features"
        lines = [f"static void tm_channel_{self.cname(channel)}({params})", "{",
                 f"    const float n = {cf(self.n)};", "    int i;"]
        decl = ["float sum = 0.0f;"]
        if need_sq:
            decl.append("float sumsq = 0.0f;")
        if need_peak:
            decl.append("float peak = 0.0f;")
        if need_minmax:
            first = f"window[{channel}]" if channel >= 0 else "tm_mag_buf[0]"
            decl += [f"float lo = {first};", "float hi = lo;"]
        lines += ["    " + d for d in decl]
        body = ["        const float x = " + x + ";", "        sum += x;"]
        if need_sq:
            body.append("        sumsq += x * x;")
        if need_peak:
            body.append("        const float ax = x < 0.0f ? -x : x;")
            body.append("        if (ax > peak) {\n            peak = ax;\n        }")
        if need_minmax:
            body.append("        if (x < lo) {\n            lo = x;\n        }")
            body.append("        if (x > hi) {\n            hi = x;\n        }")
        lines.append(f"    for (i = 0; i < {self.n}; i++) {{")
        lines += body
        lines.append("    }")
        lines.append("    const float mean = sum / n;")
        if not need_mean:
            lines.append("    (void)mean;")
        if need_m2 or need_zcr:
            lines.append("    float m2 = 0.0f;")
            if need_m4:
                lines.append("    float m4 = 0.0f;")
            if need_zcr:
                lines.append("    int crossings = 0;")
                lines.append("    int prev_negative = 0;")
            lines.append(f"    for (i = 0; i < {self.n}; i++) {{")
            lines.append(f"        const float d = {x} - mean;")
            lines.append("        const float d2 = d * d;")
            lines.append("        m2 += d2;")
            if need_m4:
                lines.append("        m4 += d2 * d2;")
            if need_zcr:
                lines.append("        const int negative = d < 0.0f;")
                lines.append("        if (i > 0 && negative != prev_negative) {\n            crossings++;\n        }")
                lines.append("        prev_negative = negative;")
            lines.append("    }")
            lines.append("    m2 = m2 / n;")
            if need_m4:
                lines.append("    m4 = m4 / n;")
        if need_sq:
            lines.append("    const float rms = tm_sqrtf(sumsq / n);")
        if spectral:
            half = self.n_fft // 2
            width = half // NUM_BANDS
            need_best = "domfreq" in kinds
            need_weighted = "centroid" in kinds
            need_bands = any(k.startswith("band") for k in kinds)
            lines += [
                f"    for (i = 0; i < {self.n}; i++) {{",
                f"        tm_fft_re[i] = {x} - mean;",
                "        tm_fft_im[i] = 0.0f;",
                "    }",
                "    for (; i < TM_FFT_SIZE; i++) {\n        tm_fft_re[i] = 0.0f;\n        tm_fft_im[i] = 0.0f;\n    }",
                "    tm_fft(tm_fft_re, tm_fft_im);",
                "    float total = 0.0f;",
            ]
            if need_weighted:
                lines.append("    float weighted = 0.0f;")
            if need_best:
                lines += ["    float best = -1.0f;", "    int best_k = 1;"]
            if need_bands:
                lines.append(f"    float band[{NUM_BANDS}] = {{0.0f, 0.0f, 0.0f, 0.0f}};")
            lines += [
                f"    for (int k = 1; k <= {half}; k++) {{",
                "        const float p = tm_fft_re[k] * tm_fft_re[k] + tm_fft_im[k] * tm_fft_im[k];",
                "        total += p;",
            ]
            if need_weighted:
                lines.append("        weighted += (float)k * p;")
            if need_best:
                lines.append("        if (p > best) {\n            best = p;\n            best_k = k;\n        }")
            if need_bands:
                lines.append(f"        band[(k - 1) / {width}] += p;")
            lines.append("    }")
            if need_best or need_weighted:
                lines.append(f"    const float bin_hz = {cf(self.rate / self.n_fft)};")
            if not (need_weighted or need_bands):
                lines.append("    (void)total;")
        for pos, kind in items:
            lines.append(f"    features[{pos}] = {self.value_expr(kind)};")
        lines.append("}")
        return "\n".join(lines) + "\n"

    def value_expr(self, kind: str) -> str:
        return {
            "mean": "mean",
            "std": "tm_sqrtf(m2)",
            "rms": "rms",
            "peak": "peak",
            "p2p": "hi - lo",
            "crest": "rms > 1e-12f ? peak / rms : 0.0f",
            "kurt": "(m2 <= 1e-12f * tm_maxf(peak * peak, 1e-12f)) ? 0.0f : m4 / (m2 * m2)",
            "zcr": f"(float)crossings / {cf(max(self.n - 1, 1))}",
            "domfreq": "(float)best_k * bin_hz",
            "centroid": "total > 0.0f ? weighted / total * bin_hz : 0.0f",
            "band0": "total > 0.0f ? band[0] / total : 0.0f",
            "band1": "total > 0.0f ? band[1] / total : 0.0f",
            "band2": "total > 0.0f ? band[2] / total : 0.0f",
            "band3": "total > 0.0f ? band[3] / total : 0.0f",
        }[kind]


class ModelEmitter:
    def __init__(self, artifact: dict[str, Any]):
        self.a = artifact
        self.p = artifact["params"]
        self.alg = artifact["algorithm"]
        self.task = artifact["task"]

    def calibration_size(self) -> int:
        if self.task != "anomaly":
            return 0
        if self.alg == "threshold_rule":
            return 3 * len(self.p["conditions"])
        return 1

    def score_doc(self) -> str:
        return {
            "threshold_rule": "how many normal standard deviations the most unusual feature is from normal",
            "isolation_forest": "anomaly score; higher means more unusual",
            "autoencoder_int8": "reconstruction error; higher means more unusual",
            "mlp_int8": "logit margin between the top two classes",
            "gradient_boosting": "margin between the top two classes (or the raw margin for two classes)",
        }.get(self.alg, "share of trees that voted for the label")

    def emit(self) -> str:
        if self.alg == "threshold_rule":
            return self.emit_rule()
        if self.alg in ("decision_tree", "random_forest", "gradient_boosting", "isolation_forest"):
            return self.emit_trees()
        if self.alg in ("mlp_int8", "autoencoder_int8"):
            return self.emit_mlp()
        raise ValueError(f"no C emitter for {self.alg}")

    # rules ------------------------------------------------------------------------------
    def emit_rule(self) -> str:
        conds = self.p["conditions"]
        nc = len(conds)
        src = [
            f"#define TM_NUM_CONDITIONS {nc}",
            c_array("int8_t", "tm_cond_feature", [c["feature"] for c in conds]),
            c_array("int8_t", "tm_cond_above", [1 if c["op"] == "gt" else 0 for c in conds]),
            c_array("float", "tm_cond_k", [c["k"] for c in conds], cf),
            c_array("float", "tm_threshold", [c["threshold"] for c in conds], cf, const=False),
            c_array("float", "tm_mean", [c["mean"] for c in conds], cf, const=False),
            c_array("float", "tm_std", [c["std"] for c in conds], cf, const=False),
            """tinyml_result_t model_predict(const float *window)
{
    float f[TINYML_NUM_FEATURES];
    tinyml_result_t r;
    model_features(window, f);
    r.label = 0;
    r.score = -3.4e38f;
    for (int c = 0; c < TM_NUM_CONDITIONS; c++) {
        const float x = f[tm_cond_feature[c]];
        const float z = tm_cond_above[c] ? (x - tm_mean[c]) / tm_std[c] : (tm_mean[c] - x) / tm_std[c];
        if (z > r.score) {
            r.score = z;
        }
        if (tm_cond_above[c] ? (x > tm_threshold[c]) : (x < tm_threshold[c])) {
            r.label = 1;
        }
    }
    return r;
}

static int32_t tm_cal_count;
static float tm_cal_mean[TM_NUM_CONDITIONS];
static float tm_cal_m2[TM_NUM_CONDITIONS];

void model_calibrate_begin(void)
{
    tm_cal_count = 0;
    for (int c = 0; c < TM_NUM_CONDITIONS; c++) {
        tm_cal_mean[c] = 0.0f;
        tm_cal_m2[c] = 0.0f;
    }
}

void model_calibrate_add(const float *window)
{
    float f[TINYML_NUM_FEATURES];
    model_features(window, f);
    tm_cal_count++;
    for (int c = 0; c < TM_NUM_CONDITIONS; c++) {
        const float x = f[tm_cond_feature[c]];
        const float delta = x - tm_cal_mean[c];
        tm_cal_mean[c] += delta / (float)tm_cal_count;
        tm_cal_m2[c] += delta * (x - tm_cal_mean[c]);
    }
}

int model_calibrate_end(void)
{
    if (tm_cal_count < TINYML_CALIBRATION_MIN_WINDOWS) {
        return -1;
    }
    for (int c = 0; c < TM_NUM_CONDITIONS; c++) {
        const float sd = tm_sqrtf(tm_cal_m2[c] / (float)tm_cal_count);
        if (sd <= 0.0f) {
            continue;
        }
        tm_mean[c] = tm_cal_mean[c];
        tm_std[c] = sd;
        tm_threshold[c] = tm_cond_above[c] ? tm_mean[c] + tm_cond_k[c] * sd : tm_mean[c] - tm_cond_k[c] * sd;
    }
    return 0;
}

void model_calibration_get(float *out)
{
    for (int c = 0; c < TM_NUM_CONDITIONS; c++) {
        out[3 * c] = tm_threshold[c];
        out[3 * c + 1] = tm_mean[c];
        out[3 * c + 2] = tm_std[c];
    }
}

void model_calibration_set(const float *in)
{
    for (int c = 0; c < TM_NUM_CONDITIONS; c++) {
        tm_threshold[c] = in[3 * c];
        tm_mean[c] = in[3 * c + 1];
        tm_std[c] = in[3 * c + 2];
    }
}
""",
        ]
        return "\n".join(src)

    # trees ------------------------------------------------------------------------------
    def emit_trees(self) -> str:
        p = self.p
        feature, left, right = p["feature"], p["left"], p["right"]
        leaves = np.asarray(p["leaves"], dtype=np.float64)
        n_leaves, out_dim = leaves.shape
        idx_type = int_type(left + right + p["roots"])
        cmp = "<=" if p["op"] == "le" else "<"
        src = [
            f"#define TM_NUM_TREES {len(p['roots'])}",
            f"#define TM_NUM_NODES {len(feature)}",
            f"#define TM_OUT_DIM {out_dim}",
            c_array(int_type(feature), "tm_node_feature", feature, per_line=16),
            c_array("float", "tm_node_threshold", p["threshold"], cf, 6),
            c_array(idx_type, "tm_node_left", left, per_line=12),
            c_array(idx_type, "tm_node_right", right, per_line=12),
            c_array(idx_type, "tm_roots", p["roots"], per_line=12),
            c_array("float", "tm_leaf", leaves.ravel().tolist(), cf, 6),
        ]
        walk = f"""static void tm_forest(const float *f, float *acc)
{{
    for (int d = 0; d < TM_OUT_DIM; d++) {{
        acc[d] = 0.0f;
    }}
    for (int t = 0; t < TM_NUM_TREES; t++) {{
        int32_t node = tm_roots[t];
        while (tm_node_feature[node] >= 0) {{
            node = (f[tm_node_feature[node]] {cmp} tm_node_threshold[node]) ? tm_node_left[node] : tm_node_right[node];
        }}
        const int32_t leaf = tm_node_right[node] * TM_OUT_DIM;
        for (int d = 0; d < TM_OUT_DIM; d++) {{
            acc[d] += tm_leaf[leaf + d];
        }}
    }}
}}
"""
        src.append(walk)
        agg = p["aggregate"]
        if agg == "depth":
            src.append(f"""static float tm_depth_threshold = {cf(p['depth_threshold'])};
static const float tm_train_mean_depth = {cf(p['train_mean_depth'])};
static const float tm_c_norm = {cf(p['c_norm'])};

static float tm_avg_depth(const float *window)
{{
    float f[TINYML_NUM_FEATURES];
    float acc[TM_OUT_DIM];
    model_features(window, f);
    tm_forest(f, acc);
    return acc[0] / (float)TM_NUM_TREES;
}}

tinyml_result_t model_predict(const float *window)
{{
    tinyml_result_t r;
    const float avg = tm_avg_depth(window);
    r.label = avg < tm_depth_threshold ? 1 : 0;
    r.score = 1.0f - avg / tm_c_norm;
    return r;
}}

static int32_t tm_cal_count;
static float tm_cal_sum;

void model_calibrate_begin(void)
{{
    tm_cal_count = 0;
    tm_cal_sum = 0.0f;
}}

void model_calibrate_add(const float *window)
{{
    tm_cal_sum += tm_avg_depth(window);
    tm_cal_count++;
}}

int model_calibrate_end(void)
{{
    if (tm_cal_count < TINYML_CALIBRATION_MIN_WINDOWS) {{
        return -1;
    }}
    tm_depth_threshold = {cf(p['depth_threshold'])} * ((tm_cal_sum / (float)tm_cal_count) / tm_train_mean_depth);
    return 0;
}}

void model_calibration_get(float *out)
{{
    out[0] = tm_depth_threshold;
}}

void model_calibration_set(const float *in)
{{
    tm_depth_threshold = in[0];
}}
""")
            return "\n".join(src)
        base = p.get("base") or [0.0] * out_dim
        add_base = ""
        if agg == "margin":
            src.append(c_array("float", "tm_base", base, cf))
            add_base = "    for (int d = 0; d < TM_OUT_DIM; d++) {\n        acc[d] += tm_base[d];\n    }\n"
        if out_dim == 1:
            decide = "    r.label = acc[0] > 0.0f ? 1 : 0;\n    r.score = acc[0];\n"
        else:
            score = "acc[best] / (float)TM_NUM_TREES" if agg == "proba" else "acc[best] - second"
            decide = f"""    int best = 0;
    for (int d = 1; d < TM_OUT_DIM; d++) {{
        if (acc[d] > acc[best]) {{
            best = d;
        }}
    }}
    float second = -3.4e38f;
    for (int d = 0; d < TM_OUT_DIM; d++) {{
        if (d != best && acc[d] > second) {{
            second = acc[d];
        }}
    }}
    (void)second;
    r.label = best;
    r.score = {score};
"""
        src.append(f"""tinyml_result_t model_predict(const float *window)
{{
    float f[TINYML_NUM_FEATURES];
    float acc[TM_OUT_DIM];
    tinyml_result_t r;
    model_features(window, f);
    tm_forest(f, acc);
{add_base}{decide}    return r;
}}
""")
        return "\n".join(src)

    # 8-bit networks -------------------------------------------------------------------------
    def emit_mlp(self) -> str:
        q = self.p
        w1 = np.asarray(q["w1"], dtype=int)
        w2 = np.asarray(q["w2"], dtype=int)
        hidden, n_in = w1.shape
        n_out = w2.shape[0]
        src = [
            f"#define TM_HIDDEN {hidden}",
            f"#define TM_OUTPUTS {n_out}",
            c_array("float", "tm_in_mean", q["in_mean"], cf, 6),
            c_array("float", "tm_in_mul", q["in_mul"], cf, 6),
            c_array("int8_t", "tm_w1", w1.ravel().tolist(), per_line=16),
            c_array("int32_t", "tm_b1", q["b1"], per_line=8),
            c_array("int8_t", "tm_w2", w2.ravel().tolist(), per_line=16),
            c_array("int32_t", "tm_b2", q["b2"], per_line=8),
            f"#define TM_M1 {int(q['m1'])}LL",
            f"#define TM_SHIFT1 {int(q['shift1'])}",
            """static int32_t tm_round_clamp(float v)
{
    if (v >= 127.0f) {
        return 127;
    }
    if (v <= -127.0f) {
        return -127;
    }
    int32_t q = (int32_t)v;
    const float frac = v - (float)q;
    if (frac >= 0.5f) {
        q += 1;
    } else if (frac <= -0.5f) {
        q -= 1;
    }
    return q;
}

static void tm_network(const float *window, int32_t *q_in, int32_t *out)
{
    float f[TINYML_NUM_FEATURES];
    int32_t q_h[TM_HIDDEN];
    model_features(window, f);
    for (int i = 0; i < TINYML_NUM_FEATURES; i++) {
        q_in[i] = tm_round_clamp((f[i] - tm_in_mean[i]) * tm_in_mul[i]);
    }
    for (int h = 0; h < TM_HIDDEN; h++) {
        int32_t acc = tm_b1[h];
        for (int i = 0; i < TINYML_NUM_FEATURES; i++) {
            acc += (int32_t)tm_w1[h * TINYML_NUM_FEATURES + i] * q_in[i];
        }
        if (acc < 0) {
            acc = 0;
        }
        int64_t v = ((int64_t)acc * TM_M1 + (1LL << (TM_SHIFT1 - 1))) >> TM_SHIFT1;
        q_h[h] = v > 127 ? 127 : (int32_t)v;
    }
    for (int o = 0; o < TM_OUTPUTS; o++) {
        int32_t acc = tm_b2[o];
        for (int h = 0; h < TM_HIDDEN; h++) {
            acc += (int32_t)tm_w2[o * TM_HIDDEN + h] * q_h[h];
        }
        out[o] = acc;
    }
}
""",
        ]
        if q["kind"] == "autoencoder":
            src.append(f"""#define TM_M2 {int(q['m2'])}LL
#define TM_SHIFT2 {int(q['shift2'])}
static float tm_err_threshold = {cf(q['err_threshold'])};
static const float tm_train_mean_err = {cf(q['train_mean_err'])};

static int64_t tm_error(const float *window)
{{
    int32_t q_in[TINYML_NUM_FEATURES];
    int32_t out[TM_OUTPUTS];
    tm_network(window, q_in, out);
    int64_t err = 0;
    for (int o = 0; o < TM_OUTPUTS; o++) {{
        int64_t v = ((int64_t)out[o] * TM_M2 + (1LL << (TM_SHIFT2 - 1))) >> TM_SHIFT2;
        if (v > 32767) {{
            v = 32767;
        }} else if (v < -32768) {{
            v = -32768;
        }}
        const int64_t e = v - q_in[o];
        err += e * e;
    }}
    return err;
}}

tinyml_result_t model_predict(const float *window)
{{
    tinyml_result_t r;
    const int64_t err = tm_error(window);
    r.label = (float)err > tm_err_threshold ? 1 : 0;
    r.score = (float)err;
    return r;
}}

static int32_t tm_cal_count;
static float tm_cal_sum;

void model_calibrate_begin(void)
{{
    tm_cal_count = 0;
    tm_cal_sum = 0.0f;
}}

void model_calibrate_add(const float *window)
{{
    tm_cal_sum += (float)tm_error(window);
    tm_cal_count++;
}}

int model_calibrate_end(void)
{{
    if (tm_cal_count < TINYML_CALIBRATION_MIN_WINDOWS || tm_train_mean_err <= 0.0f) {{
        return -1;
    }}
    tm_err_threshold = {cf(q['err_threshold'])} * ((tm_cal_sum / (float)tm_cal_count) / tm_train_mean_err);
    return 0;
}}

void model_calibration_get(float *out)
{{
    out[0] = tm_err_threshold;
}}

void model_calibration_set(const float *in)
{{
    tm_err_threshold = in[0];
}}
""")
        else:
            if n_out == 1:
                decide = "    r.label = out[0] > 0 ? 1 : 0;\n    r.score = (float)out[0] * TM_SCORE_SCALE;\n"
            else:
                decide = """    int best = 0;
    for (int o = 1; o < TM_OUTPUTS; o++) {
        if (out[o] > out[best]) {
            best = o;
        }
    }
    int32_t second = INT32_MIN;
    for (int o = 0; o < TM_OUTPUTS; o++) {
        if (o != best && out[o] > second) {
            second = out[o];
        }
    }
    r.label = best;
    r.score = (float)(out[best] - second) * TM_SCORE_SCALE;
"""
            src.append(f"""#define TM_SCORE_SCALE {cf(q['score_scale'])}

tinyml_result_t model_predict(const float *window)
{{
    int32_t q_in[TINYML_NUM_FEATURES];
    int32_t out[TM_OUTPUTS];
    tinyml_result_t r;
    tm_network(window, q_in, out);
{decide}    return r;
}}
""")
        return "\n".join(src)


def generate(artifact: dict[str, Any], title: str) -> Generated:
    env = Environment(loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined, trim_blocks=True,
                      lstrip_blocks=True, keep_trailing_newline=True)
    features = FeatureEmitter(artifact)
    model = ModelEmitter(artifact)
    window = artifact["window"]
    ctx = {
        "title": title,
        "version": __version__,
        "algorithm": artifact["algorithm"],
        "quantization": artifact["quantization"],
        "task": artifact["task"],
        "axes": window["axes"],
        "window_samples": window["samples"],
        "num_axes": len(window["axes"]),
        "rate_c": cf(window["sample_rate_hz"]),
        "hop": window["hop"],
        "num_features": len(artifact["features"]),
        "num_classes": len(artifact["classes"]),
        "classes": artifact["classes"],
        "feature_names": [f["name"] for f in artifact["features"]],
        "score_doc": model.score_doc(),
        "calibration_size": model.calibration_size(),
        "feature_code": features.emit(),
        "needs_maxf": any(f["kind"] == "kurt" for f in artifact["features"]),
        "needs_sqrt": any(f["kind"] in ("std", "rms", "crest") or f["channel_index"] < 0 for f in artifact["features"])
        or artifact["algorithm"] == "threshold_rule",
        "model_code": model.emit(),
        "c_string": c_string,
        "explanation": artifact.get("explanation", ""),
    }
    return Generated(
        header=env.get_template("tinyml_model.h.j2").render(**ctx),
        source=env.get_template("tinyml_model.c.j2").render(**ctx),
        cpp=env.get_template("TinyMLModel.hpp.j2").render(**ctx),
        example=env.get_template("example_main.c.j2").render(**ctx),
    )


def write(generated: Generated, dest: Path) -> dict[str, Path]:
    dest.mkdir(parents=True, exist_ok=True)
    paths = {
        "header": dest / "tinyml_model.h",
        "source": dest / "tinyml_model.c",
        "cpp": dest / "TinyMLModel.hpp",
        "example": dest / "example_main.c",
    }
    paths["header"].write_text(generated.header)
    paths["source"].write_text(generated.source)
    paths["cpp"].write_text(generated.cpp)
    paths["example"].write_text(generated.example)
    return paths
