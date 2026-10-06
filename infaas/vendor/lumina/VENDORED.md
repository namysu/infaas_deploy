# Vendored from Lumina (new_podexecutor)

Source: `/home/test/ysnam/new_podexecutor` @ `8eb588c63b8533618b42f4d5acfa580a015bb7b5`.
The files are byte-for-byte copies. **Do not edit them.** Re-copy from a new commit
and update this file instead.

| file | source | why |
|---|---|---|
| `models.py` | `podexec/common/models.py` | same 43-model set and short-name ↔ HF id mapping (PLAN A-03) |
| `dali_preprocess.py` | `podexec/worker/dali_preprocess.py` | same preprocessing path and per-GPU `PREPROCESS_MODE` (PLAN A-04) |
| `preload_libs.py` | `podexec/worker/preload_libs.py` | same library preload at worker start (PLAN A-05) |

`infaas/proto/podexec.proto` is a copy of `podexec/proto/podexec.proto` from the same
commit. It exists so the INFaaS front-end can serve the Lumina wire API (PLAN §3.4).
