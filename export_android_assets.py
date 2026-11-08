"""Export everything the SANKOFA Android app needs, per language subset:

  * the vetted answer bank (training questions, their answer-cluster ids, cluster answers)
  * mined scaffolds (registers), self-tuned view weights
  * confidence models (strict = exact vetted answer, lenient = exact or chrF++ >= 50)
  * thresholds calibrated on the full validation split (answer-or-escalate lambda, conformal q-hat)
  * the validation split (question + gold answer) so the phone can re-run the evaluation
  * Python reference predictions, used to check that the Java engine matches this code

Usage:  python3 export_android_assets.py <out_dir>
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sankofa.paths import DATA
from sankofa.retrieve import SankofaRetriever, VIEWS_ALL
from sankofa.fusion import learn_weights
from sankofa.metrics import akey, chrf
from sankofa.abstain import feat_matrix, ConfidenceModel, ltt_threshold, FEATS

OUT = sys.argv[1] if len(sys.argv) > 1 else "android_assets"
os.makedirs(OUT, exist_ok=True)
tr, va = pd.read_csv(DATA + "Train.csv"), pd.read_csv(DATA + "Val.csv")
cfg = dict(norm=True, scaffold=True, views=VIEWS_ALL)
LANG_NAMES = {"Eng_Uga": "English (Uganda)", "Lug_Uga": "Luganda (Uganda)", "Eng_Ken": "English (Kenya)",
              "Swa_Ken": "Swahili (Kenya)", "Eng_Eth": "English (Ethiopia)", "Amh_Eth": "Amharic (Ethiopia)",
              "Eng_Gha": "English (Ghana)", "Aka_Gha": "Akan (Ghana)"}


def model_json(cm):
    if cm.const is not None:
        return {"const": cm.const}
    return {"coef": cm.m.coef_[0].tolist(), "intercept": float(cm.m.intercept_[0])}


def conformal_qhat(Fn, gold, alpha=0.2):
    sc = np.where(gold >= 0, 1 - Fn[np.arange(len(gold)), np.maximum(gold, 0)], np.inf)
    n = len(sc); k = int(np.ceil((n + 1) * (1 - alpha)))
    return None if k > n or not np.isfinite(np.sort(sc)[k - 1]) else float(np.sort(sc)[k - 1])


manifest = []
for s in sorted(tr.subset.unique()):
    lang = s[:3]
    T, V = tr[tr.subset == s], va[va.subset == s]
    Q, A = T.input.tolist(), T.output.tolist()
    w, _, dev = learn_weights(lang, Q, A, cfg)
    R = SankofaRetriever(lang, fusion="weighted", weights=w, **cfg).fit(Q, A)
    fused, sc = R.score(V.input.tolist())
    top, feats = R.predict(None, sc=sc)
    preds = [R.cluster_answer[c] for c in top]
    exact = np.array([akey(p) == akey(g) for p, g in zip(preds, V.output)])
    cf = np.array([chrf(p, g) for p, g in zip(preds, V.output)])
    Xd, Xv = feat_matrix(dev["feats"], w), feat_matrix(feats, w)
    cm_strict = ConfidenceModel().fit(Xd, dev["exact"].astype(float))
    cm_len = ConfidenceModel().fit(Xd, (dev["exact"] | (dev["chrf"] >= 50)).astype(float))
    conf_s, conf_l = cm_strict.predict(Xv), cm_len.predict(Xv)
    lam_s = ltt_threshold(conf_s, ~exact)
    lam_l = ltt_threshold(conf_l, ~(exact | (cf >= 50)))
    idx = {akey(a): i for i, a in enumerate(R.cluster_answer)}
    gold = np.array([idx.get(akey(a), -1) for a in V.output])
    Fn = fused / np.maximum(fused.max(1, keepdims=True), 1e-9)
    qhat = conformal_qhat(Fn, gold)
    d = {
        "subset": s, "lang": lang, "name": LANG_NAMES[s],
        "weights": w, "feature_order": FEATS,
        "scaffolds": {"suffix": [a for a, _, _ in R.sc.get("suffix", [])],
                      "prefix": [a for a, _, _ in R.sc.get("prefix", [])],
                      "openers": sorted(R.sc.get("openers", []))},
        "bank": {"questions": Q, "q2c": R.q2c.tolist(), "answers": R.cluster_answer},
        "confidence": {"strict": model_json(cm_strict), "lenient": model_json(cm_len)},
        "thresholds": {"lambda_strict": None if not np.isfinite(lam_s) else float(lam_s),
                       "lambda_lenient": None if not np.isfinite(lam_l) else float(lam_l),
                       "conformal_qhat": qhat, "alpha": 0.2, "delta": 0.1},
        "validation": {"questions": V.input.tolist(), "gold": V.output.tolist()},
        "reference": {"top": top.tolist(), "conf_strict": np.round(conf_s, 6).tolist(),
                      "fused_top": np.round(feats["top"], 6).tolist(),
                      "acc": float(exact.mean() * 100), "chrf": float(cf.mean())},
    }
    with open(os.path.join(OUT, f"{s}.json"), "w", encoding="utf8") as f:
        json.dump(d, f, ensure_ascii=False, separators=(",", ":"))
    manifest.append({"subset": s, "name": LANG_NAMES[s], "file": f"{s}.json",
                     "n_bank": len(Q), "n_clusters": R.nc, "n_val": len(V),
                     "ref_acc": round(d["reference"]["acc"], 2), "ref_chrf": round(d["reference"]["chrf"], 2)})
    print(s, "acc", round(exact.mean() * 100, 2), "chrF", round(cf.mean(), 2), "lambda", lam_s, lam_l, "qhat", qhat, flush=True)
with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)
