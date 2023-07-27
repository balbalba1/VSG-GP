import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gpflow
from gpflow.ci_utils import reduce_in_tests
from gpflow.config import (
    default_float,
    set_default_float,
    set_default_summary_fmt,
)
from gpflow.utilities import ops, print_summary
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from IPython.core.display import HTML
from sklearn.decomposition import PCA
set_default_float(np.float64)
set_default_summary_fmt("notebook")

Y_raw = np.loadtxt(r"C:\Users\26950\Desktop\GPLVM-code\GMM-VSG-main\data\23items copy.txt")
Y_raw = Y_raw[0:23,1:7]
Y_raw = tf.convert_to_tensor(Y_raw, dtype=default_float())

labels = np.loadtxt(r"C:\Users\26950\Desktop\GPLVM-code\GMM-VSG-main\data\23items copy.txt")
labels = labels[0:23,0:1]
labels = tf.convert_to_tensor(labels,dtype=default_float())

#standard
s_scale = StandardScaler()
s_scale.fit(Y_raw)
Y= s_scale.transform(Y_raw)

print(
    "Number of points: {} and Number of dimensions: {}".format(
        Y.shape[0], Y.shape[1]
    )
)

#model construction
latent_dim = 2  # number of latent dimensions
num_inducing = 0  # number of inducing pts    
num_data = Y.shape[0]  # number of data points
#initialize via PCA
X_mean_init = ops.pca_reduce(Y, latent_dim)
X_var_init = tf.ones((num_data, latent_dim), dtype=default_float())
#Pick inducing inputs randomly from dataset initialization
np.random.seed(1)  # for reproducibility
inducing_variable = tf.convert_to_tensor(
    np.random.permutation(X_mean_init.numpy())[:num_inducing],
    dtype=default_float(),
)
#SE kernel
lengthscales = tf.convert_to_tensor([0.1] * latent_dim, dtype=default_float())
kernel = gpflow.kernels.RBF(lengthscales=lengthscales)

gplvm = gpflow.models.BayesianGPLVM(
    Y,
    X_data_mean=X_mean_init,
    X_data_var=X_var_init,
    kernel=kernel,
    inducing_variable=inducing_variable,    
)
# Instead of passing an inducing_variable directly, we can also set the num_inducing_variables argument to an integer, which will randomly pick from the data.

#change the default likelihood variance,which is 1,to 0.01
gplvm.likelihood.variance.assign(0.01)
#optimize the created model,given that this model has a determinisitic evidence lower bound,use SciPy's BFGS to optimize
opt = gpflow.optimizers.Scipy()
maxiter = reduce_in_tests(1000)
opt.minimize(
    gplvm.training_loss,
    method="BFGS",
    variables=gplvm.trainable_variables,

)
gplvm_X_mean = gplvm.X_data_mean.numpy()
gplvm_X_var=gplvm.X_data_var.numpy()
X=gplvm_X_mean
X1=X.transpose()
inv=np.linalg.inv(np.dot(X1,X))
W=np.dot(inv,np.dot(X1,Y)).transpose()

# Model analysis
print(labels)
#sample
X_mean=np.mean(X,axis=0)
rng=check_random_state(0)

var=np.var(X,axis=0)
cov=var*np.eye(2)
samples_var=[]
for i in range(X.shape[0]):
    Xnew_samples_var=rng.multivariate_normal(X[i],cov,14)
    samples_var.append(Xnew_samples_var)
samples_var = np.asarray(samples_var)
samples_var=samples_var.reshape(-1,2)
VY_separate_var = np.dot(samples_var,W.transpose())
VY_X_separate_var= s_scale.inverse_transform(VY_separate_var)

#nonlinear fitting
data=np.genfromtxt(r"C:\Users\26950\Desktop\GPLVM-code\GMM-VSG-main\data\23items copy.txt")
X1 = data[:, 1:7]
Y1 = data[:, 0].reshape(-1, 1)

model = gpflow.models.GPR(
    (X1, Y1),
    kernel=gpflow.kernels.SquaredExponential(),
)
m=model

opt = gpflow.optimizers.Scipy()
opt.minimize(m.training_loss,m.trainable_variables)

Xnew_var=VY_X_separate_var
mean_var,var_1=model.predict_f(Xnew_var)
created_data_var=np.hstack((Xnew_var,mean_var))
np.savetxt('result-VSG_VY_X_separate_var.csv',created_data_var,delimiter=' ',fmt='%.2f')