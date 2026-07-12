# Integration Provenance

This repository was assembled from two audited source snapshots:

- Core UAV-FGS release baseline: `abdaa34ab65b50e1b33f9865833e7692109f9d02`
- Reference-depth evaluation tools: `831a01a42e998c059b26538b17e4118d645d83cb`

Integration changes are intentionally narrow:

- restored public upstream attribution, download, and XMP namespace URLs that had
  been replaced in the release snapshot;
- added optional `--train_list` and `--test_list` loading support required by the
  geometry protocol; both default to empty strings and preserve the baseline
  split behavior when omitted;
- added the depth-reference evaluation tools and their `numba` dependency;
- removed machine-specific paths, obsolete pilot launchers, generated artifacts,
  and internal drafting notes.

No submitted experimental result files are embedded in this source repository.
