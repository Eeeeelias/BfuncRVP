from torch.utils.data import Dataset

# defining a new simulated Dataset
class Sim_Dataset(Dataset):
    '''
    n_patients = number of patients for which the phenotype is simulated, default is 160 000 as it was in the FuncRVP preprint
    n_genes = number of tested genes for which the beta and functional embedding should be simulated, default 20 000 as usually is
    n_dim = dimension of the functional embedding, defualt 256 to which the func. embedding was reduced to in FuncRVP preprint
    '''
    def __init__(self, n_patients = 160000, n_genes = 20000, n_dim = 256):
        pass
    def simulate_phenotype(self, n_patients):
        pass
    def simulate_gt_betas(self, n_genes):
        pass
    def create_func_embeddings(self, n_genes, n_dim):
        pass

