import math
import torch
import anndata
import scanpy as sc
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.metrics import r2_score
from scipy import sparse
import anndata
import numpy as np
import scanpy as sc
from scipy import sparse
import seaborn as sns
import pandas as pd
from train import ConditionalFlowModel
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import issparse
from adjustText import adjust_text
from scipy.stats import spearmanr
import argparse
import scanpy as sc
sc.settings.figdir = "./umap_plots"  
os.makedirs(sc.settings.figdir, exist_ok=True)

def compute_mmd(x, y, kernel='rbf', gamma=1.0):
    """Compute Maximum Mean Discrepancy (MMD)"""
    from sklearn.metrics.pairwise import pairwise_kernels
    xx = pairwise_kernels(x, x, metric=kernel, gamma=gamma)
    yy = pairwise_kernels(y, y, metric=kernel, gamma=gamma)
    xy = pairwise_kernels(x, y, metric=kernel, gamma=gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()



# ===== Dataset Class =====
class FlowMatchDataset(Dataset):
    def __init__(self, x_ctrl, x_pert, cell_type):
        self.x_ctrl = x_ctrl
        self.x_pert = x_pert
        self.cell_type = cell_type

    def __len__(self):
        return len(self.x_ctrl)

    def __getitem__(self, idx):
        return {
            'x_ctrl': self.x_ctrl[idx],
            'x_pert': self.x_pert[idx],
            'cell_type': self.cell_type[idx]
        }

def evaluate(test_pt_path="test_covid.pt", model_path="cfm_model.pt", out_prefix="", save_dir="./results/"):
    subplot_data = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Load test dataset
    test_data = torch.load(test_pt_path)
    x_ctrl = test_data['x_ctrl']
    x_pert = test_data['x_pert']
    cell_types = test_data['cell_type']
    gene_names = test_data.get('gene_names', [f"Gene{i}" for i in range(x_ctrl.shape[1])])
    cell_type_mapping = test_data['cell_type_mapping']

    if isinstance(cell_type_mapping, dict):
        rev_cell_type_map = {v: k for k, v in cell_type_mapping.items()}
    elif isinstance(cell_type_mapping, (np.ndarray, list)):
        rev_cell_type_map = {i: name for i, name in enumerate(cell_type_mapping)}
    else:
        raise TypeError("Unsupported type for cell_type_mapping")

    unique_cell_types = torch.unique(cell_types)
    input_dim = x_ctrl.shape[1]

    model = ConditionalFlowModel(input_dim=input_dim, num_cell_types=len(cell_type_mapping))
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    results = []
    sp_list = []
    mmd_list = []

    for ct_id in unique_cell_types:
        idx = (cell_types == ct_id)
        ct_name = rev_cell_type_map[int(ct_id)]

        x_ctrl_ct = x_ctrl[idx].to(device)
        x_pert_ct = x_pert[idx].to(device)
        cell_type_ct = cell_types[idx].to(device)

        y_pred = x_ctrl_ct.clone()
        n_steps = 10
        for t_val in torch.linspace(0, 1, n_steps):
            t = t_val.view(1, 1).repeat(y_pred.size(0), 1).to(device)
            v_t = model(y_pred, t, x_ctrl_ct, cell_type_ct)
            y_pred += v_t * (1.0 / n_steps)

        pred_np = y_pred.detach().cpu().numpy()
        x_ctrl_np = x_ctrl_ct.cpu().numpy()
        x_pert_np = x_pert_ct.cpu().numpy()

        pred_avg = np.mean(pred_np, axis=0)
        real_avg = np.mean(x_pert_np, axis=0)

        # MMD calculation
        mmd_score = compute_mmd(pred_np, x_pert_np)
        mmd_list.append((ct_name, mmd_score))

        # Spearman correlation
        spearman_corr, _ = spearmanr(pred_avg, real_avg)
        sp_list.append((ct_name, spearman_corr))

        # R2 for all genes and top 100 DEGs
        all_data = anndata.AnnData(np.concatenate([x_ctrl_np, pred_np, x_pert_np], axis=0))
        all_data.obs['condition'] = ['ctrl']*x_ctrl_np.shape[0] + ['pred']*pred_np.shape[0] + ['real']*x_pert_np.shape[0]
        all_data.var_names = gene_names
        sc.tl.rank_genes_groups(all_data, groupby="condition", method="wilcoxon")

        degs = all_data.uns["rank_genes_groups"]["names"]["real"]
        top_100_deg_idx = [np.where(all_data.var_names == gene)[0][0] for gene in degs[:100]]

        r2_all = np.corrcoef(pred_avg, real_avg)[0, 1] ** 2
        r2_top = np.corrcoef(pred_avg[top_100_deg_idx], real_avg[top_100_deg_idx])[0, 1] ** 2

        results.append({
            "cell_type": ct_name,
            "R2_all_genes": r2_all,
            "R2_top100_DEGs": r2_top
        })

    df_r2 = pd.DataFrame(results)
    df_mmd = pd.DataFrame(mmd_list, columns=["cell_type", "MMD"])
    df_spearman = pd.DataFrame(sp_list, columns=["cell_type", "Spearman_corr"])
    df_r2.to_csv(os.path.join(save_dir, f"{out_prefix}_r2.csv"), index=False)
    df_mmd.to_csv(os.path.join(save_dir, f"{out_prefix}_mmd.csv"), index=False)
    df_spearman.to_csv(os.path.join(save_dir, f"{out_prefix}_spearman.csv"), index=False)

    print(df_r2)
    print(df_mmd)
    print(df_spearman)

    return df_r2, df_mmd, df_spearman


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Evaluate ConditionalFlowModel on test set and save metrics separately")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test .pt file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model .pt file")
    parser.add_argument("--save_dir", type=str, default="./results/", help="Directory to save evaluation results")
    parser.add_argument("--out_prefix", type=str, default="CFM", help="Prefix for output files (default: 'CFM')")
    args = parser.parse_args()

    evaluate(
        test_pt_path=args.test_path,
        model_path=args.model_path,
        out_prefix=args.out_prefix,
        save_dir=args.save_dir
    )
