import numpy as np
import scanpy as sc
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

adata = # path to h5ad file
condition_key = "condition"
cell_type_key = "celltype"
ctrl_key = "control"
pert_key = "severe COVID-19"
min_cells = 0

output_path = "CFM/train_covid_gene.pt"

# ==== Step 1: Filter cell types with enough data ====
summary = adata.obs.groupby([cell_type_key, condition_key]).size().unstack(fill_value=0)
valid_cell_types = summary[
    (summary[ctrl_key] >= min_cells) & (summary[pert_key] >= min_cells)
].index.tolist()

print(f"Using {len(valid_cell_types)} valid cell types with enough control and perturbed cells.")

# ==== Step 2: Collect paired samples ====
paired_data = []

for cell_type in valid_cell_types:
    adata_ctrl = adata[(adata.obs[cell_type_key] == cell_type) & (adata.obs[condition_key] == ctrl_key)]
    adata_pert = adata[(adata.obs[cell_type_key] == cell_type) & (adata.obs[condition_key] == pert_key)]
    
    # Match by minimum number of cells
    min_n = min(adata_ctrl.shape[0], adata_pert.shape[0])
    X_ctrl = adata_ctrl.X[:min_n].toarray() if not isinstance(adata_ctrl.X, np.ndarray) else adata_ctrl.X[:min_n]
    X_pert = adata_pert.X[:min_n].toarray() if not isinstance(adata_pert.X, np.ndarray) else adata_pert.X[:min_n]
    
    paired_data.append({
        'x_ctrl': X_ctrl,
        'x_pert': X_pert,
        'cell_type': [cell_type] * min_n
    })

# ==== Step 3: Merge and encode ====
Xc_all = np.vstack([d['x_ctrl'] for d in paired_data])
Xp_all = np.vstack([d['x_pert'] for d in paired_data])
cell_types_all = sum([d['cell_type'] for d in paired_data], [])

# Encode cell types
le = LabelEncoder()
cell_type_encoded = le.fit_transform(cell_types_all)

# Convert to tensors
Xc_tensor = torch.tensor(Xc_all, dtype=torch.float32)
Xp_tensor = torch.tensor(Xp_all, dtype=torch.float32)
cell_type_tensor = torch.tensor(cell_type_encoded, dtype=torch.long)

print(f"Final dataset shape: {Xc_tensor.shape} (samples, genes)")
print(f"Number of unique cell types: {len(le.classes_)}")
print(Xp_tensor.shape)



# Check if gene names are present
if hasattr(adata, "var_names") and adata.var_names is not None:
    gene_names = adata.var_names.tolist()
    print(f"Gene names detected. Sample: {gene_names[:10]}")
else:
    gene_names = [f"Gene{i}" for i in range(Xc_tensor.shape[1])]
    print("No gene names found. Using default gene names.")



# ==== Print Detailed Statistics ====
print("\n==== Dataset Statistics ====")
print(f"Total samples (cells): {Xc_tensor.shape[0]}")
print(f"Number of genes (features): {Xc_tensor.shape[1]}")
print(f"Number of unique cell types: {len(le.classes_)}")
print(f"Unique cell types: {list(le.classes_)}")
print(f"Control samples shape: {Xc_tensor.shape}")
print(f"Perturbed samples shape: {Xp_tensor.shape}")


ctrl_counts = {ct: len(d['x_ctrl']) for d, ct in zip(paired_data, valid_cell_types)}
pert_counts = {ct: len(d['x_pert']) for d, ct in zip(paired_data, valid_cell_types)}

print("\nNumber of paired samples per cell type:")
for ct in valid_cell_types:
    print(f"  - {ct}: {ctrl_counts[ct]} control, {pert_counts[ct]} perturbed")

