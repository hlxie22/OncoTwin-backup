# Molecular Graph and Explanation Engine

## Purpose

The molecular graph layer gives the digital twin a mechanism-aware personalization and explanation system. It maps biomarkers and molecular features into pathway-level representations that modify mechanistic parameters and explain why the model behaves a certain way.

The goal is not to overclaim molecular causality. The goal is to make missing biomarkers, subtype biology, and pathway-level uncertainty visible in the twin.

## What the molecular graph should do

The graph should support four functions:

1. Encode patient molecular/pathology features into a mechanism embedding.
2. Modify mechanistic parameter priors, such as proliferation or drug sensitivity.
3. Explain which biological pathways influenced the twin.
4. Rank missing biomarkers by expected impact on uncertainty.

## Graph structure

### Node types

```text
gene nodes:
  BRCA1, BRCA2, TP53, PIK3CA, ESR1, ERBB2, PTEN, PALB2, AKT1, RB1

pathway nodes:
  DNA repair
  estrogen receptor signaling
  HER2 signaling
  PI3K/AKT/mTOR
  cell cycle
  apoptosis
  immune microenvironment
  angiogenesis / vascular delivery

drug/mechanism nodes:
  anthracycline sensitivity
  taxane sensitivity
  platinum sensitivity
  endocrine sensitivity
  HER2-targeted sensitivity
  immune checkpoint sensitivity

phenotype nodes:
  proliferation
  invasion
  drug resistance
  residual disease risk
  toxicity vulnerability
```

### Edge types

```text
gene participates_in pathway
mutation affects pathway
pathway influences phenotype
phenotype modifies mechanistic parameter
drug targets mechanism
missing biomarker increases uncertainty
```

## Patient graph features

Each patient gets node features such as:

```text
known mutated
known wildtype
unknown/missing
expression high/low if available
copy-number gain/loss if available
pathology proxy present
subtype proxy present
```

Example:

```typescript
type MolecularProfile = {
  brca1?: "mutated" | "wildtype" | "unknown";
  brca2?: "mutated" | "wildtype" | "unknown";
  hrd?: "positive" | "negative" | "unknown";
  tp53?: "mutated" | "wildtype" | "unknown";
  pik3ca?: "mutated" | "wildtype" | "unknown";
  esr1?: "mutated" | "wildtype" | "unknown";
  erbb2?: "amplified" | "not_amplified" | "unknown";
};
```

## First implementation: rules + embeddings

Start simple. Use curated rules to produce parameter modifiers and explanations. These same rules seed the **biology-informed residual prior** the parameter amortizer learns against (family E in `06_ai_personalization_parameter_amortizer.md`): they set the center of each parameter's prior, and the amortizer learns only the patient-specific deviation.

Example rules:

```text
High Ki-67 or grade 3 → increase proliferation prior.
BRCA/HRD positive → increase DNA-repair sensitivity modifier and reduce uncertainty if known.
BRCA/HRD unknown → increase uncertainty in DNA-repair-related treatment sensitivity.
HER2 positive → activate HER2 signaling node and relevant pathway explanation.
ER positive → activate ER signaling node and endocrine-sensitivity explanation.
TNBC → downweight ER/HER2 mechanisms and emphasize chemotherapy response, proliferation, DNA repair, and immune uncertainty.
```

Output:

```typescript
type MolecularRuleOutput = {
  parameterModifiers: {
    proliferationShift: number;
    dnaRepairSensitivityShift: number;
    endocrineSensitivityShift: number;
    her2SensitivityShift: number;
    immuneResponseShift: number;
  };
  explanations: string[];
  missingDataDrivers: string[];
};
```

## Second implementation: graph attention network

Once the app has enough structured training data, add a graph neural network.

Architecture:

```text
patient molecular features
        ↓
node feature initialization
        ↓
graph attention layers
        ↓
pathway embeddings
        ↓
parameter modifier head
        ↓
explanation/attention output
```

Output:

```typescript
type MolecularGraphOutput = {
  mechanismEmbedding: number[];
  pathwayAttention: {
    pathway: string;
    attentionWeight: number;
    explanation: string;
  }[];
  parameterModifiers: {
    proliferationShift: number;
    dnaRepairSensitivityShift: number;
    endocrineResistanceShift: number;
    immuneResponseShift: number;
    vascularDeliveryShift: number;
  };
  missingDataValue: {
    biomarker: string;
    expectedUncertaintyReduction: number;
    explanation: string;
  }[];
};
```

## How graph output modifies the simulator

The AI parameter amortizer gives base parameter distributions. The molecular graph modifies them.

Example:

```text
base proliferation prior
  + pathology modifier
  + molecular graph modifier
  = final proliferation prior
```

Pseudocode:

```python
def apply_molecular_modifiers(param_dist, graph_output):
    param_dist.proliferation.mean *= exp(graph_output.proliferation_shift)
    param_dist.drug_sensitivity["platinum"].mean *= exp(graph_output.dna_repair_sensitivity_shift)
    param_dist.endocrine_sensitivity.mean *= exp(graph_output.endocrine_sensitivity_shift)
    return param_dist
```

## Missing-data handling

Unknown data should be represented explicitly.

Example:

```text
BRCA/HRD unknown does not mean negative.
It means the DNA-repair sensitivity modifier has wider uncertainty.
```

The app should output:

```text
DNA-repair sensitivity is uncertain because BRCA/HRD status is missing. Adding this result would reduce uncertainty in drug-sensitivity assumptions.
```

## Explanation format

Use two levels of explanation.

### Patient-facing explanation

```text
The model places more weight on proliferation because this case has high-grade disease and high Ki-67. DNA-repair sensitivity remains uncertain because BRCA/HRD status is missing.
```

### Research-facing explanation

```text
High attention nodes: cell cycle, TP53, DNA repair, angiogenesis. These shifted proliferation and chemotherapy-sensitivity priors and widened uncertainty around residual disease.
```

## Guardrails

Do not say:

```text
This mutation means you will respond to treatment.
```

Say:

```text
This biomarker changes the model's assumptions about one biological pathway, but it does not determine treatment response by itself.
```

## Training data

Use:

```text
TCGA-BRCA for molecular/subtype graph pretraining
METABRIC for breast cancer molecular-clinical associations
I-SPY2/related trial metadata where biomarkers overlap with response labels
```

The molecular graph may be partly curated because not all desired markers appear consistently across imaging datasets.

## Implementation milestones

1. Curated molecular rules.
2. Missingness-aware biomarker flags.
3. Parameter modifier interface.
4. Explanation generator.
5. Molecular graph schema.
6. Graph embedding pretraining on TCGA-BRCA/METABRIC-style data.
7. Graph attention model integration.
8. Attention-to-explanation translation.
9. Value-of-information ranking for missing biomarkers.
