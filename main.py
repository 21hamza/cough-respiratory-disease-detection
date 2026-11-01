"""Cough Sound-Based Respiratory Disease Detection: complete pipeline.

Runs every step that produced the results in the final report, from raw Coswara audio to the final figures:

    python main.py                          # everything (about 2.5 h on a 2-core CPU)
    python main.py --stage train            # one stage (see STAGES below)
    python main.py --from-stage tune        # this stage and all later ones
    python main.py --stage train --models m2 m3      # restrict model stages to some models

Stages, in order:
    download   fetch 15 Coswara batches and extract the heavy-cough recordings      src/data/download_coswara.py
    dataset    label, clean, batch-match, split and cache features (MFCC, log-Mel) src/data/build_dataset.py
    yamnet     download pretrained YAMNet and cache its embeddings (for M4)         src/data/extract_yamnet.py
    tune       hyperparameter grid + nested 5-fold cross-validation, all models     src/run_experiment.py --cv
    train      train the selected setting of each model on the fixed split,
               evaluate on the test set and save the trained models                 src/run_experiment.py
    analyse    out-of-fold error analysis with bootstrap confidence intervals       src/error_analysis.py
    robust     noise-robustness test of the trained models                          src/robustness.py
    figures    comparison tables and all report figures                             src/make_figures.py
Every stage writes its console output to logs/.
"""
import argparse, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ["m1", "m2", "m3", "m4"]
STAGES = {
    "download": (["-m", "src.data.download_coswara"], False),
    "dataset":  (["-m", "src.data.build_dataset"], False),
    "yamnet":   (["-m", "src.data.extract_yamnet"], False),
    "tune":     (["-m", "src.run_experiment", "--cv"], True),
    "train":    (["-m", "src.run_experiment"], True),
    "analyse":  (["-m", "src.error_analysis"], False),
    "robust":   (["-m", "src.robustness"], False),
    "figures":  (["-m", "src.make_figures"], False),
}


def run_stage(name, models):
    args, takes_models = STAGES[name]
    cmd = [sys.executable, "-u", *args] + (["--models", *models] if takes_models else [])
    log = ROOT / "logs" / f"{name}.log"
    log.parent.mkdir(exist_ok=True)
    print(f"\n{'=' * 78}\n[{datetime.now():%H:%M:%S}] STAGE {name.upper()}   ->  {' '.join(cmd[1:])}\n{'=' * 78}", flush=True)
    t0 = time.time()
    with open(log, "w") as f, subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as p:
        for line in p.stdout:
            if "Warning" in line or "warnings.warn" in line or "XNNPACK" in line:
                continue
            print(line, end="", flush=True)
            f.write(line)
    if p.returncode != 0:
        sys.exit(f"stage '{name}' failed (exit code {p.returncode}); see {log}")
    print(f"[{datetime.now():%H:%M:%S}] stage {name} finished in {(time.time() - t0) / 60:.1f} min  (log: {log.relative_to(ROOT)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--stage", choices=STAGES, help="run only this stage")
    g.add_argument("--from-stage", choices=STAGES, help="run this stage and all later ones")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS, help="models for the tune/train stages")
    a = ap.parse_args()
    names = list(STAGES)
    todo = [a.stage] if a.stage else names[names.index(a.from_stage):] if a.from_stage else names
    t0 = time.time()
    for s in todo:
        run_stage(s, a.models)
    print(f"\nAll done in {(time.time() - t0) / 60:.1f} min. Results: results/   Figures: figures/   Logs: logs/")


if __name__ == "__main__":
    main()
