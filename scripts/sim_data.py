from torch.utils.data import Dataset
import numpy as np

# defining a new simulated Dataset
class Sim_Dataset(Dataset):
    '''
    n_patients = number of patients for which the phenotype is simulated, default is 2 000 
    n_genes = number of tested genes for which the beta and functional embedding should be simulated, default 20 000 as usually is
    n_dim = dimension of the functional embedding, defualt 256 to which the func. embedding was reduced to in FuncRVP preprint
    '''
    def __init__(self, n_patients = 2000, n_genes = 20000, n_dim = 256):
        self.betas = self.simulate_gt_betas(n_genes)
        self.g_matrix = self.simulate_g_matrix(n_patients, n_genes)
        self.phenos = self.compute_phenotype()
    def simulate_gt_betas(self, n_genes):
        abs_beta = 5
        repeat = 5
        betas = np.array([abs_beta]*repeat + [-abs_beta]*repeat + list(np.random.normal(0, 1, n_genes - 2*repeat)))
        return betas

    def simulate_g_matrix(self, n_patients, n_genes):
        p_variant = 0.3
        g_matrix = np.random.binomial(1, p_variant, size=(n_genes, n_patients))
        return g_matrix

    def compute_phenotype(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        phenotype_raw = np.dot(self.betas, self.g_matrix)
        phenotype_binary = (sigmoid(phenotype_raw) >= 0.5).astype(int)
        return phenotype_binary

    def create_func_embeddings(self, n_genes, n_dim):
        pass

n_data = Sim_Dataset()
