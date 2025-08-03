import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import os


# ======================
# Dataset Class
# ======================

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


# ======================
# Model Definition
# ======================


class ConditionalFlowModel(nn.Module):
    def __init__(self, input_dim, num_cell_types, hidden_dim=256):
        super().__init__()
        self.cell_embed = nn.Embedding(num_cell_types, 16)
        self.time_embed = nn.Sequential(nn.Linear(1, 16), nn.ReLU())
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2 + 16 + 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, xt, t, x_ctrl, cell_type):
        cell_emb = self.cell_embed(cell_type)
        t_emb = self.time_embed(t)
        h = torch.cat([xt, x_ctrl, cell_emb, t_emb], dim=-1)
        return self.net(h)


# ======================
# Training Function
# ======================

def train_flow_matching(model, train_loader, val_loader, num_epochs=100, lr=1e-3, device='cuda', save_path='model.pt'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            x_ctrl = batch['x_ctrl'].to(device)
            x_pert = batch['x_pert'].to(device)
            cell_type = batch['cell_type'].to(device)

            t = torch.rand(x_ctrl.size(0), 1).to(device)
            xt = (1 - t) * x_ctrl + t * x_pert
            v_target = x_pert - x_ctrl

            v_pred = model(xt, t, x_ctrl, cell_type)
            loss = loss_fn(v_pred, v_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                x_ctrl = batch['x_ctrl'].to(device)
                x_pert = batch['x_pert'].to(device)
                cell_type = batch['cell_type'].to(device)

                t = torch.rand(x_ctrl.size(0), 1).to(device)
                xt = (1 - t) * x_ctrl + t * x_pert
                v_target = x_pert - x_ctrl

                v_pred = model(xt, t, x_ctrl, cell_type)
                val_loss += loss_fn(v_pred, v_target).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    return model


# ======================
# Main Training Script
# ======================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ConditionalFlowModel using .pt datasets")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training .pt file")
    parser.add_argument("--val_path", type=str, required=True, help="Path to the validation .pt file")
    parser.add_argument("--save_path", type=str, default="model.pt", help="Path to save the trained model")
    parser.add_argument("--epoch", type=str, default=50)
    parser.add_argument("--lr", type=str, default=1e-3)
    args = parser.parse_args()

    # Load datasets
    train_data = torch.load(args.train_path)
    val_data = torch.load(args.val_path)

    train_dataset = FlowMatchDataset(train_data['x_ctrl'], train_data['x_pert'], train_data['cell_type'])
    val_dataset = FlowMatchDataset(val_data['x_ctrl'], val_data['x_pert'], val_data['cell_type'])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = ConditionalFlowModel(input_dim=train_data['x_ctrl'].shape[1],
                                 num_cell_types=len(train_data['cell_type_mapping']))

    train_flow_matching(model, train_loader, val_loader, num_epochs=int(args.epoch),lr=float(args.lr), save_path=args.save_path)


