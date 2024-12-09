from firthlogist import FirthLogisticRegression
import sim_data as sd

def perform_firths(dataset):
    fl = FirthLogisticRegression()
    fl.fit(dataset.g_matrix.T, dataset.phenos)
    print(fl.summary())


dataset1 = sd.SimulatedDataset(random_seed=42, n_genes=2000, n_patients=200)
perform_firths(dataset1)