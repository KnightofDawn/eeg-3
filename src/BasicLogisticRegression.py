import numpy as np
import lib.EegData as eeg
from sklearn.linear_model import LogisticRegression

train_x = eeg.load_folder('e:/eeg/data/train_bin/', eeg.EegData.TYPE_DATA)
train_y = eeg.load_folder('e:/eeg/data/train_bin/', eeg.EegData.TYPE_EVENTS)

train_x = dict(zip([x.name for x in train_x], train_x))
train_y = dict(zip([x.name for x in train_y], train_y))

labels = train_x.keys()
train_labels = labels[2:]
test_labels = labels[0:2]

train_x_data = [train_x[label].data for label in train_labels]
train_y_data = [train_y[label].data for label in train_labels]

test_x_data = [train_x[label].data for label in test_labels]
test_y_data = [train_y[label].data for label in test_labels]

train_x_all = np.concatenate(train_x_data)
train_y_all = np.concatenate(train_y_data)

test_x_all = np.concatenate(test_x_data)
test_y_all = np.concatenate(test_y_data)

print train_x_all.shape, train_y_all.shape, test_x_all.shape, test_y_all.shape

model = LogisticRegression()
model.fit(train_x_all, train_y_all[:, 0])
print model.score(test_x_all, test_y_all[:, 0])
