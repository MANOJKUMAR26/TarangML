Owner: Dev A. CSV reading, column summaries, column roles, data checks, resampling, windows.

- The assistant only ever sees `ColumnSummary` objects. Never pass rows to `decide()`.
- Every data issue has a code in `DEFAULT_SEVERITY`, a plain-English message with numbers, and a next step.
- Windows never cross a gap; splits are by recording (see `src/tinyml/ml/split.py`).
