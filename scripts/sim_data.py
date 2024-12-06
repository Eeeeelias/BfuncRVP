from torch.utils.data import Dataset

# defining a new simulated Dataset
class Sim_Dataset(Dataset):
    def __init__(self, n_patients = 160000, n_genes = 20000, n_dim = 256):
        pass
    def simulate_phenotype(self, n_patients):
        pass
    def simulate_gt_betas(self, n_genes):
        pass
    def create_func_embeddings(self, n_genes, n_dim):
        pass

