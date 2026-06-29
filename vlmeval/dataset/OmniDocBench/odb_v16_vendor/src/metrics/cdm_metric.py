import logging
import os
import shutil

from .cdm.cdm import cdm_metrics


def _should_save_cdm_vis() -> bool:
    return os.getenv("CDM_SAVE_VIS", "1").lower() not in {"0", "false", "no"}


def _zero_metrics(error_message: str = "") -> dict[str, float | int | str]:
    metrics: dict[str, float | int | str] = {
        "recall": 0.0,
        "precision": 0.0,
        "F1_score": 0.0,
        "tp": 0,
        "gt_tokens": 0,
        "pred_tokens": 0,
    }
    if error_message:
        metrics["cdm_eval_error"] = error_message
    return metrics

class CDM:
    def __init__(self, output_root="./result"):
        self.output_root = output_root

    def _legacy_output_path_map(self, img_id):
        img_id = str(img_id)
        return {
            "gt_bbox_jsonl": os.path.join(self.output_root, "gt", "bbox", f"{img_id}.jsonl"),
            "gt_base_png": os.path.join(self.output_root, "gt", "vis", f"{img_id}_base.png"),
            "gt_vis_png": os.path.join(self.output_root, "gt", "vis", f"{img_id}.png"),
            "pred_bbox_jsonl": os.path.join(self.output_root, "pred", "bbox", f"{img_id}.jsonl"),
            "pred_base_png": os.path.join(self.output_root, "pred", "vis", f"{img_id}_base.png"),
            "pred_vis_png": os.path.join(self.output_root, "pred", "vis", f"{img_id}.png"),
            "match_base_png": os.path.join(self.output_root, "vis_match", f"{img_id}_base.png"),
            "match_png": os.path.join(self.output_root, "vis_match", f"{img_id}.png"),
        }

    def _persist_legacy_outputs(self, cdm_vis, img_id):
        if not isinstance(cdm_vis, dict):
            return
        for key, dst in self._legacy_output_path_map(img_id).items():
            src = cdm_vis.get(key)
            if not src or not os.path.exists(src):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

        case_dir = cdm_vis.get("case_dir")
        if case_dir and os.path.isdir(case_dir):
            shutil.rmtree(case_dir, ignore_errors=True)

    def _cleanup_legacy_outputs(self, img_id):
        for path in self._legacy_output_path_map(img_id).values():
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

    def _evaluate_sample(self, gt_latex, pred_latex, img_id, sample_context=None):
        try:
            return cdm_metrics(
                gt_latex,
                pred_latex,
                save_vis=_should_save_cdm_vis(),
                tmp_dir=self.output_root,
                persist_vis_dir=os.path.join(self.output_root, "_cdm_cases"),
                vis_name=str(img_id),
            )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            context_parts = []
            if isinstance(sample_context, dict):
                for key in ("img_id", "gt_idx", "pred_idx"):
                    value = sample_context.get(key)
                    if value is not None:
                        context_parts.append(f"{key}={value}")
            context_suffix = f" ({', '.join(context_parts)})" if context_parts else ""
            logging.exception("CDM sample failed for %s%s: %s", img_id, context_suffix, exc)
            self._cleanup_legacy_outputs(img_id)
            return _zero_metrics(error_message=error_message)

    def evaluate(self, gt_latex, pred_latex, img_id, sample_context=None):
        os.makedirs(self.output_root, exist_ok=True)
        metrics = self._evaluate_sample(gt_latex, pred_latex, img_id, sample_context=sample_context)
        if _should_save_cdm_vis():
            self._persist_legacy_outputs(metrics.get("cdm_vis"), img_id)
        metrics.pop("cdm_vis", None)
        metrics.pop("cdm_vis_error", None)
        return metrics
