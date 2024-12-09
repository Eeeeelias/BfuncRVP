import sklearn.metrics
from firthlogist import FirthLogisticRegression
import sim_data as sd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def perform_firths(dataset): # doesn't work much let's face it lost cause
    fl = FirthLogisticRegression()
    fl.fit(dataset.g_matrix.T, dataset.phenos)
    print(fl.summary())

def perform_lin_reg_reg(dataset, penalty='l1'): # regularized logistic regression
    lasso_model = LogisticRegression(penalty=penalty, solver='liblinear')
    lasso_model.fit(dataset.g_matrix.T, dataset.phenos)
    print(lasso_model.coef_)
    print(dataset.betas)
    return(lasso_model.coef_[0])



dataset1 = sd.SimulatedDataset(random_seed=42, n_genes=2000, n_patients=200)
#perform_firths(dataset1)
lasso_betas = perform_lin_reg_reg(dataset1, penalty='l2')
MSE = sklearn.metrics.mean_squared_error(dataset1.betas, lasso_betas)
print(MSE)