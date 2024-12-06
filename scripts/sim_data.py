from torch.utils.data import Dataset
import numpy as np

# defining a new simulated Dataset
class Sim_Dataset(Dataset):
    '''
    n_patients = number of patients for which the phenotype is simulated, default is 160 000 as it was in the FuncRVP preprint
    p_pheno = probability of p(disease), lower than 0.5?
    n_genes = number of tested genes for which the beta and functional embedding should be simulated, default 20 000 as usually is
    n_dim = dimension of the functional embedding, defualt 256 to which the func. embedding was reduced to in FuncRVP preprint
    '''
    def __init__(self, n_patients = 160000, p_pheno = .3,  n_genes = 20000, n_dim = 256):
        self.phenos = self.simulate_phenotype(n_patients, p_pheno)

    # simulating functions
    def simulate_phenotype(self, n_patients, p_pheno):
        phenos = np.random.binomial(1, p=p_pheno, size=n_patients)
        return phenos

    def simulate_gt_betas(self, n_genes):
        pass

    def create_func_embeddings(self, n_genes, n_dim):
        pass


n_data = Sim_Dataset()
