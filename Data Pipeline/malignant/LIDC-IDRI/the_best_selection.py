"""
the_best_selection.py
---------------------
Selects the best training images from clean_malignant/ and copies them
to the_BEST/ using the data-optimisation rules for low-probability images.

Selection rules (in priority order):
  INCLUDE if cnn_score >= 0.35                    (clear positive signal)
  INCLUDE if cnn_score < 0.35 AND readers >= 5    (hard nodule, many readers confirm it)
  EXCLUDE if cnn_score < 0.35 AND readers <= 2    (weak label AND no detectable signal)
           AND nodule_contrast < 0

Rationale for low-probability images:
  - Sub-GGO nodules (HU -850 to -700) are real malignancies; excluding by HU alone
    removes exactly the subtle cases a clinical CNN needs to detect.
  - High reader count (>= 5) is a strong signal that the nodule is genuinely there
    even if the HU is faint — keep these even if the CNN score is low.
  - The only safe exclusion: <= 2 readers AND negative contrast AND low score.
    This combination means the image has no detectable signal and weak annotation
    support — it adds noise, not learning signal.

Usage:
    python the_best_selection.py
    python the_best_selection.py --eval /tmp/eval_v2.json --src /path/to/clean_malignant
                                 --dst /path/to/the_BEST

Outputs:
    the_BEST/patient-XXXX/nodule_N.dcm    (selected images)
    the_BEST/excluded.csv                  (excluded images with reason)
    the_BEST/selection_report.json         (full selection log)
"""

import os
import csv
import json
import shutil
import argparse


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_EVAL = "/tmp/eval_v2.json"
DEFAULT_SRC  = "/path/to/malignant working/clean malignant"
DEFAULT_DST  = "/path/to/malignant working/the_BEST"


# ── Selection logic ───────────────────────────────────────────────────────────

def decide(score, readers, contrast):
    """
    Apply the three-rule selection logic for low-probability images.

    Returns:
        (keep: bool, reason: str)
    """
    if score >= 0.35:
        return True, f"score={score:.3f} >= 0.35"

    if readers >= 5:
        return True, f"score={score:.3f} < 0.35 but readers={readers} >= 5 (confirmed hard nodule)"

    if readers <= 2 and contrast < 0:
        return False, (f"score={score:.3f}, readers={readers} <= 2, "
                       f"contrast={contrast:.1f} < 0 — no detectable signal, weak label")

    # score < 0.35 but readers 3-4, or contrast >= 0 — borderline keep
    return True, f"score={score:.3f}, readers={readers}, contrast={contrast:.1f} — borderline kept"


# ── Main ──────────────────────────────────────────────────────────────────────

def run_selection(eval_path=DEFAULT_EVAL, src=DEFAULT_SRC, dst=DEFAULT_DST):
    """
    Load evaluation JSON, apply selection rules, copy selected images to dst/.
    """
    with open(eval_path) as f:
        evals = json.load(f)

    included = []
    excluded = []

    for e in evals:
        patient  = e['patient']
        nodule   = e['nodule']
        score    = e.get('cnn_score', 0.0)
        readers  = e.get('readers', 0)
        contrast = e.get('nodule_contrast', 0.0)

        keep, reason = decide(score, readers, contrast)

        src_path  = os.path.join(src, patient, f"nodule_{nodule}.dcm")
        dest_path = os.path.join(dst, patient, f"nodule_{nodule}.dcm")

        entry = {
            'patient':          patient,
            'nodule':           nodule,
            'cnn_score':        score,
            'readers':          readers,
            'nodule_contrast':  contrast,
            'ntype':            e.get('ntype', 'unknown'),
            'keep':             keep,
            'reason':           reason,
        }

        if keep:
            included.append({**entry, 'src': src_path, 'dest': dest_path})
        else:
            excluded.append(entry)

    # Copy included files
    copied, errors = 0, []
    for item in included:
        if not os.path.exists(item['src']):
            errors.append(f"Missing: {item['src']}")
            continue
        os.makedirs(os.path.dirname(item['dest']), exist_ok=True)
        shutil.copy2(item['src'], item['dest'])
        copied += 1

    # Write excluded.csv
    excl_path = os.path.join(dst, 'excluded.csv')
    with open(excl_path, 'w', newline='') as f:
        fields = ['patient', 'nodule', 'cnn_score', 'readers', 'nodule_contrast', 'ntype', 'reason']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in excluded:
            w.writerow({k: e[k] for k in fields})

    # Write full selection report
    report_path = os.path.join(dst, 'selection_report.json')
    with open(report_path, 'w') as f:
        json.dump({'included': included, 'excluded': excluded}, f, indent=2)

    # Summary
    print(f"\n{'='*50}")
    print(f"Total evaluated:  {len(evals)}")
    print(f"Included:         {copied}")
    print(f"Excluded:         {len(excluded)}")
    if errors:
        print(f"Copy errors:      {len(errors)}")

    print(f"\nExcluded images:")
    for e in excluded:
        print(f"  {e['patient']} n{e['nodule']}  "
              f"score={e['cnn_score']:.3f}  readers={e['readers']}  "
              f"contrast={e['nodule_contrast']:.1f}  — {e['reason']}")

    # Score distribution of included images
    scores = [i['cnn_score'] for i in included]
    if scores:
        import statistics
        print(f"\nIncluded CNN score stats:")
        print(f"  mean={statistics.mean(scores):.3f}  "
              f"median={statistics.median(scores):.3f}  "
              f"min={min(scores):.3f}  max={max(scores):.3f}")
        tiers = {
            'Excellent (>=0.80)': sum(s >= 0.80 for s in scores),
            'Good (0.55-0.80)':   sum(0.55 <= s < 0.80 for s in scores),
            'Fair (0.35-0.55)':   sum(0.35 <= s < 0.55 for s in scores),
            'Poor (<0.35, kept)': sum(s < 0.35 for s in scores),
        }
        for label, count in tiers.items():
            pct = 100 * count / len(scores)
            print(f"  {label}: {count} ({pct:.1f}%)")

    print(f"\nFiles saved to: {dst}")
    print(f"Excluded log:   {excl_path}")
    print(f"Full report:    {report_path}")
    return included, excluded


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Select best training images from clean_malignant/')
    parser.add_argument('--eval', default=DEFAULT_EVAL,
                        help='Path to eval_v2.json from evaluate.py')
    parser.add_argument('--src',  default=DEFAULT_SRC,
                        help='Source folder (clean malignant/)')
    parser.add_argument('--dst',  default=DEFAULT_DST,
                        help='Destination folder (the_BEST/)')
    args = parser.parse_args()
    run_selection(args.eval, args.src, args.dst)
