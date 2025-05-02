# DeepSignalingFlow Data 
Input data description of the network DeepSignalingFlow 


### Drug Datasets

- **DrugComb**  
  The results of drug combination screening studies for a large variety of cancer cell lines.

- **DrugCombDB**  
  Similar to DrugComb but includes more raw data.

- **NCI Almanac**  
  Standardized dataset developed by the National Cancer Institute (NCI) to support research into anticancer drug combinations. Provides detailed synergy measurements for drug pairs across a panel of cancer cell lines.

- **O’Neil**  
  High-quality experimental dataset of drug combinations developed by O’Neil et al. (2016).

#### Columns Used from All Drug Datasets

| Column     | Meaning                                                         |
|------------|-----------------------------------------------------------------|
| DrugA      | The name of the first drug in the combination.                 |
| DrugB      | The name of the second drug in the combination.                |
| Cell_line  | The cancer cell line where the drug combination was tested.    |
| Score      | Synergy score.                                                 |

---

### Gene Datasets

- **KEGG Pathway**  
  A collection of manually drawn pathway maps representing molecular interaction, reaction, and relation networks across:
  - Metabolism
  - Genetic Information Processing
  - Environmental Information Processing
  - Cellular Processes
  - Organismal Systems
  - Human Diseases
  - Drug Development  
  
  48 cancer-related signaling pathways are extracted, and all genes involved in these pathways are used.

- **DrugBank**  
  Offers comprehensive and reliable drug data. Each drug in the synergy datasets (Drug A, Drug B) is mapped to its known gene targets using DrugBank.

- **Cell Model Passports**  
  Contains over 2000 catalogued cancer cell line models. For each cell line in the drug datasets, two omics data types are extracted:
  
  - **RNA-Seq Expression**  
    Quantitative gene expression data obtained using RNA sequencing (RNA-Seq).  
    *Simple explanation:* Measures which genes are active in a cell and how active they are.
  
  - **Copy Number Variation (CNV)**  
    Indicates differences in the number of copies of a gene or DNA segment.  
    *Simple explanation:* Shows whether parts of the genome are amplified (extra copies) or deleted in cancer cells.

---

### How Input Graphs Are Formed

#### Nodes

- **Gene Nodes:**
  - Feature 1: RNA-Seq expression value
  - Feature 2: Copy number variation (CNV)
  - Feature 3: Targeted by Drug A? (0 or 1)
  - Feature 4: Targeted by Drug B? (0 or 1)

- **Drug Nodes:**
  - Feature 1: 0
  - Feature 2: 0
  - Feature 3: 0
  - Feature 4: 0

#### Edges

- **Gene-Gene Connections**: Extracted from KEGG dataset.
- **Drug-Gene Connections**: Bidirectional; extracted from DrugBank.

These edges form the **adjacency matrix** \( A \in \mathbb{R}^{n \times n} \) for the graph.

---

### Resulting Matrices

- **A** — Adjacency matrix (includes gene–gene and drug–gene connections)  
- **D_in** — In-degree diagonal matrix  
- **D_out** — Out-degree diagonal matrix  
- **X** — Node feature matrix (n × 4)
