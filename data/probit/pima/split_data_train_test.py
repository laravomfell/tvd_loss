
import numpy as np

# We set the random seed

np.random.seed(1)

# We load the data

data = np.loadtxt('data.txt', delimiter = ",")
n = data.shape[ 0 ]

# We generate the training test splits

n_splits = 50
for i in range(n_splits):

    permutation = np.random.choice(range(n), n, replace = False)

    end_train = int(round(n * 9.0 / 10))
    end_test = n

    index_train = permutation[ 0 : end_train ]
    index_test = permutation[ end_train : n ]

    np.savetxt("index_train_{}.txt".format(i+1), index_train, fmt = '%d')
    np.savetxt("index_test_{}.txt".format(i+1), index_test, fmt = '%d')

    print i

np.savetxt("n_splits.txt", np.array([ n_splits ]), fmt = '%d')

# We store the index to the features and to the target

index_features = np.array(range(data.shape[ 1 ] - 1), dtype = int)
index_target = np.array([ data.shape[ 1 ] - 1 ])

np.savetxt("index_features.txt", index_features, fmt = '%d')
np.savetxt("index_target.txt", index_target, fmt = '%d')
