"""The integer square root used on chips without hardware sqrt must be correctly rounded."""

import re
import shutil
import subprocess

import numpy as np
import pytest

from tinyml.codegen.emit import TEMPLATES


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_soft_sqrt_matches_ieee(tmp_path):
    template = (TEMPLATES / "tinyml_model.c.j2").read_text()
    body = re.search(r"#else\n/\* Correctly rounded.*?\n(static float tm_sqrtf\(float x\)\n\{.*?\n\})\n#endif", template, re.S)
    assert body, "soft sqrt not found in template"
    rng = np.random.default_rng(1)
    values = np.concatenate([
        rng.uniform(0, 1, 20000), rng.uniform(0, 1e6, 20000), 10.0 ** rng.uniform(-40, 38, 20000),
        np.array([1e-45, 1.17549435e-38, 2.0, 4.0, 0.25, 3.4e38]),
    ]).astype(np.float32)
    values.tofile(tmp_path / "in.bin")
    src = tmp_path / "t.c"
    src.write_text("#include <stdint.h>\n#include <stdio.h>\n" + body.group(1) + """
int main(void) {
    FILE *in = fopen("in.bin", "rb"), *out = fopen("out.bin", "wb");
    float x;
    while (fread(&x, sizeof x, 1, in) == 1) { float r = tm_sqrtf(x); fwrite(&r, sizeof r, 1, out); }
    fclose(in); fclose(out); return 0;
}
""")
    subprocess.run(["cc", "-O2", str(src), "-o", str(tmp_path / "t")], check=True)
    subprocess.run([str(tmp_path / "t")], cwd=tmp_path, check=True)
    got = np.fromfile(tmp_path / "out.bin", dtype=np.float32)
    want = np.sqrt(values)
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))
