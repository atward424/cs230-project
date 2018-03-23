
from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd

tr = pd.read_csv('rf_train_preds.csv')
val = pd.read_csv('rf_val_preds.csv')
ts = pd.read_csv('rf_test_preds.csv')


tr_labs = pd.read_csv('data/train_outcomes.csv')
val_labs = pd.read_csv('data/val_outcomes.csv')
ts_labs = pd.read_csv('data/test_outcomes.csv')

print roc_auc_score(tr_labs['disc.24.hr'], tr)
print roc_auc_score(val_labs['disc.24.hr'], val)
print roc_auc_score(ts_labs['disc.24.hr'], ts)