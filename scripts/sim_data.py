from torch.utils.data import Dataset
import numpy as np

class SimulatedDataset(Dataset):
    """
    :param n_patients: number of patients for which the phenotype is simulated, default is 2 000
    :param n_genes: number of tested genes for which the beta and functional embedding should be simulated, default 20 000 as usually is
    :param n_dim: dimension of the functional embedding, default 256 to which the func. embedding was reduced to in FuncRVP preprint
    :param repeat: how many genes should have the same positive and negative beta and share embedding dimension (2*repeat)
    """

    def __init__(self, n_patients = 2000, n_genes = 20000, n_dim = 256, repeat = 5, random_seed = None):
        if random_seed:
            np.random.seed(random_seed)
        self.__n_patients = n_patients
        self.n_genes = n_genes
        self.repeat = repeat
        self.betas = self.simulate_gt_betas(n_genes)
        self.g_matrix = self.simulate_g_matrix(n_patients, n_genes)
        self.phenos = self.compute_phenotype()
        self.f_embedding = self.create_func_embeddings(n_genes, n_dim)

    def __len__(self):
        return self.__n_patients

    def __getitem__(self, idx):
        return self.g_matrix[:, idx], self.f_embedding, self.phenos[idx]

    def simulate_gt_betas(self, n_genes, abs_beta=5):
        """
        Simulate the betas for the genes
        :param n_genes: number of genes for which the betas should be simulated
        :param abs_beta: absolute value of the positive and negative
        :return: betas of shape (n_genes,)
        """
        betas = np.array([abs_beta]*self.repeat +
                         [-abs_beta]*self.repeat +
                         list(np.random.normal(0, 1, n_genes - 2*self.repeat)))
        return betas

    def simulate_g_matrix(self, n_patients, n_genes):
        """
        Simulate the gene matrix with a given number of patients and genes
        :param n_patients: number of patients for which the gene matrix should be simulated
        :param n_genes: number of genes for which the gene matrix should be simulated
        :return: gene matrix of shape (n_genes, n_patients)
        """
        p_variant = 0.3
        g_matrix = np.random.binomial(1, p_variant, size=(n_genes, n_patients))
        return g_matrix

    def compute_phenotype(self):
        """
        Compute the binary phenotype based on the g_matrix and betas
        :return: binary phenotype of shape (n_patients,)
        """
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        phenotype_raw = np.dot(self.betas, self.g_matrix)
        phenotype_binary = (sigmoid(phenotype_raw) >= 0.5).astype(int)
        return phenotype_binary

    def create_func_embeddings(self, n_genes, n_dim, n_shared_dim=75):
        """
        :param n_genes: number of genes for which the functional embeddings should be simulated
        :param n_dim: dimension of the functional embeddings
        :param n_shared_dim: how many dimensions should be shared between the first genes
        :return: functional embeddings of shape (n_genes, n_dim)
        """
        shared = np.random.normal(0, 1, n_shared_dim)
        func_emb = np.random.normal(0, 1, size=(n_genes, n_dim))
        func_emb[:self.repeat*2, :n_shared_dim] = shared

        return func_emb
