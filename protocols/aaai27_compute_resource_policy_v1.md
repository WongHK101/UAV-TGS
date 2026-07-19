# UAV-TGS AAAI-27 Compute Resource Policy v1

Policy ID: `uav-tgs-aaai27-compute-resource-policy-v1`

This policy governs temporary use of AutoDL hosts 900 and 901. It changes
where a frozen job may execute, not the formal split, recipe, camera,
reference, radiometry, or metric protocol.

## 1. Storage authority and execution roles

1. Host 900 is the authoritative AutoDL storage node for the project.
2. Host 901 is temporary, revocable scratch compute and is not a long-term
   data source.
3. No scene, method, or configuration is permanently assigned to either host.
4. Each phase may choose a host from current GPU availability, free disk,
   existing assets, transfer cost, expected runtime, and environment
   compatibility.
5. Paired internal configurations for one scene should run on the same host
   when practical to reduce host confounding in efficiency comparisons. This
   preference is not a hard constraint.
6. Performance metrics remain comparable across hosts under the frozen
   protocol. Wall-clock comparisons retain host receipts, and very small
   cross-host timing differences do not support strong conclusions.

## 2. Formal job preflight

Before every formal job starts, its immutable preflight receipt records:

- execution host;
- GPU model and UUID;
- driver and CUDA versions;
- environment hash;
- code commit;
- input hashes;
- available disk;
- whether the GPU is idle.

A job may use transferred assets only after their manifest and SHA-256
verification succeeds. Availability of host 901 never authorizes a change to
the formal split, recipe, camera, reference, radiometry, or metric protocol.

## 3. Return-to-900 and cleanup rules

1. Every formal endpoint, metric, receipt, failure artifact, and critical log
   produced on 901 is returned to 900 when its scene or job completes.
2. Corresponding 901 intermediates may be deleted only after 900 receives the
   files and verifies their count, size, and SHA-256 identities.
3. No final or irreplaceable result may exist only on 901.
4. If 901 lacks space, no new job starts. Completed assets are returned to
   900, verified, and only then may safely copied checkpoints, renders, and
   caches be removed before work resumes.
5. Every project file on 901 must remain below the isolated root
   `/root/autodl-tmp/UAV-TGS-901`.
6. Credentials, personal files, and unrelated data must not be copied to 901.
   SDKs, environments, and external source copied there are project assets and
   remain subject to final cleanup.

At project closeout, all remaining formal assets on 901 are returned to 900,
900 completeness is verified, and every project file inside the isolated 901
root is removed without touching any path outside it. A
`901_cleanup_receipt.json` records the cleanup root, pre-cleanup file count and
byte count, and post-cleanup verification result.

## 4. Machine-readable identity

The normative machine-readable companion is
`protocols/aaai27_compute_resource_policy_v1.json`. Its
`policy_markdown_sha256` binds this Markdown file. Its
`policy_payload_sha256` is the canonical JSON SHA-256 after removing only that
self-hash field.
