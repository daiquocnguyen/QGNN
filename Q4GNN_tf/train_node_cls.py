import time
import tensorflow as tf

from utils_node_cls import *
from models_node_cls import *

# Set random seed
seed = 123
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

# Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_float('learning_rate', 0.05, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_integer('hidden_size', 16, 'Hidden_size//4 = number of quaternion units within each hidden layer.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 20, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('fold', 2, 'From 0 to 9.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = \
    load_data_new_split(FLAGS.dataset, '../splits/' + FLAGS.dataset + '_split_0.6_0.2_'+ str(FLAGS.fold) + '.npz')

# quaternion preprocess for feature vectors
def quaternion_preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    features = features.todense()
    features = np.tile(features, 4) # A + Ai + Aj + Ak
    features = sp.coo_matrix(features)
    return sparse_to_tuple(features)

# Some preprocessing
features = quaternion_preprocess_features(features)
adj = preprocess_adj(adj)

tf.compat.v1.disable_eager_execution()
# Define placeholders
placeholders = {
    'adj': tf.compat.v1.sparse_placeholder(tf.float32, shape=tf.constant(adj[2], dtype=tf.int64)),
    'features': tf.compat.v1.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.compat.v1.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.compat.v1.placeholder(tf.int32),
    'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.compat.v1.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = Q4GNN(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.compat.v1.Session()

# Define model evaluation function
def evaluate(features, adj, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, adj, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.compat.v1.global_variables_initializer())

cost_val = []
# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, adj, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, adj, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(features, adj, y_test, test_mask, placeholders)
print("Test set results at final epoch:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
