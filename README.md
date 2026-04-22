# CFM-GP: Unified Conditional Flow Matching to Learn Gene Perturbation Across Cell Types

## Overview

Understanding gene perturbation effects across diverse cellular contexts is a central challenge in functional genomics, with significant implications for therapeutic discovery and precision medicine. While single-cell technologies enable high-resolution measurement of transcriptional responses, collecting such data remains expensive and time-intensive, especially when repeated for each cell type. Existing computational methods attempt to predict these responses but typically require separate models per cell type, limiting scalability and generalization.

CFM-GP (**C**onditional **F**low **M**atching for **G**ene **P**erturbation) is a novel deep learning framework that learns a continuous, time-dependent transformation between unperturbed and perturbed gene expression distributions, conditioned on cell type. This allows a single model to predict the transcriptional effect of a perturbation across all cell types, eliminating the need for cell type–specific training. CFM-GP employs the **flow matching objective** to model perturbation dynamics in a scalable manner.

---

## Key Features

- **Cell Type–Agnostic Prediction**: Learns perturbation effects across all cell types via a single model, with no need for cell type–specific retraining.
- **Continuous Trajectory Modeling**: Utilizes a vector field formulation to learn time-dependent perturbation trajectories.
- **Generalization Across Contexts**: Transfers perturbation knowledge across datasets and species.
- **Biological Fidelity**: Recovers pathway-level signals validated via enrichment analysis.

---

## CFM-GP Framework
![CFM-GP Framework](cfm.png)

---

## Installation and Environment Setup

```bash
git clone https://github.com/abrarrahmanabir/CFM-GP.git
cd CFM-GP
pip install -r requirements.txt
```

📂 Dataset Access

The five fully processed datasets used in this study can be downloaded from:

👉 https://drive.google.com/file/d/1sJxHM4te1CNShBLUrLVEGPrkEbOjM7mk/view?usp=sharing

After downloading, extract and place into:

./data/
🧬 Data Processing Pipeline
Data Sources



