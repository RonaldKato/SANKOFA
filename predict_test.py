"""Fit SANKOFA on Train+Val and write test predictions (ID, output, confidence, decision).
Usage: python3 predict_test.py <data_dir> <out_csv>"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sankofa.retrieve import SankofaRetriever, VIEWS_ALL
from sankofa.fusion import learn_weights
from sankofa.abstain import feat_matrix, ConfidenceModel, ltt_threshold

D, OUT = sys.argv[1], sys.argv[2]
tr, va, te = (pd.read_csv(os.path.join(D, f)) for f in ("Train.csv", "Val.csv", "Test.csv"))
cfg = dict(norm=True, scaffold=True, views=VIEWS_ALL)
rows = []
for s in sorted(te.subset.unique()):
    lang = s[:3]
    T, V, X = tr[tr.subset == s], va[va.subset == s], te[te.subset == s]
    # 1) calibrate the escalation threshold on Val with a Train-only system
    w, _, dev = learn_weights(lang, T.input.tolist(), T.output.tolist(), cfg)
    R = SankofaRetriever(lang, fusion="weighted", weights=w, **cfg).fit(T.input.tolist(), T.output.tolist())
    cm = ConfidenceModel().fit(feat_matrix(dev["feats"], w), dev["exact"].astype(float))
    top, f = R.predict(V.input.tolist())
    wrong = np.array([R.cluster_answer[c].split() != a.split() for c, a in zip(top, V.output)])
    lam = ltt_threshold(cm.predict(feat_matrix(f, w)), wrong)
    # 2) final bank = Train + Val, same weights / confidence model
    TV = pd.concat([T, V])
    R2 = SankofaRetriever(lang, fusion="weighted", weights=w, **cfg).fit(TV.input.tolist(), TV.output.tolist())
    top, f = R2.predict(X.input.tolist())
    conf = cm.predict(feat_matrix(f, w))
    for i, c, p in zip(X.ID, top, conf):
        rows.append(dict(ID=i, output=R2.cluster_answer[c], confidence=round(float(p), 4),
                         decision="answer" if p >= lam else "escalate"))
    print(s, "lambda", lam, "answer rate", float(np.mean(conf >= lam)).__round__(3), flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
