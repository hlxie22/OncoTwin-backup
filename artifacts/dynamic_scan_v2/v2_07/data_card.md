# OncoTwin V2 data card

## Final supervised fitting population

- Rows: 17,194
- Patients: 2,443
- Eligibility: original CHORD train/val and `survival_mask=True`
- Original CHORD test used in V2-07 fit: no
- External outcomes used in V2-07 fit: no

The final head reuses frozen upstream representations prepared under earlier checkpoints. Genomics must satisfy the strict availability-before-landmark rule inherited from the frozen pipeline.
