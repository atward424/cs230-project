import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('outputs/train_metrics_list.pkl', 'rb') as f:
    train = pickle.load(f)
with open('outputs/val_metrics_list.pkl', 'rb') as f:
    val = pickle.load(f)

trau = [train[i]['auroc'] for i in range(len(train))]
vau = [val[i]['auroc'] for i in range(len(train))]
# import pdb; pdb.set_trace()
fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
ax.set_color_cycle(['red', 'blue'])
ax.plot(np.arange(1, 301), trau[:300], label = 'Train')
ax.plot(np.arange(1, 301), vau[:300], label = 'Val')
ax.set_ylim(0,1)
ax.set_xlabel('epoch')
ax.set_ylabel('AUROC')
ax.set_title('FC Net Training Curves')
ax.legend()

fig.savefig('outputs/train_curve.png')