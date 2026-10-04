# Record realism notes - v4

Design references used for structure, not copied branding or text:
- Johns Hopkins breast pathology educational material: patient identifiers, accession/date fields, clinical history, specimen/body site, diagnosis, gross description, pathologist identity, and ER/PR/HER2 special studies.
- Public sample CBC reports: dense patient/specimen identity block, result/flag/unit/reference-range columns, verification metadata, and simple monochrome tabulation.
- Public hospital/radiology samples: clinical history/indication, technique, comparison, findings, impression, and report-status/version metadata.

Authenticity choices:
- no common polished design system across departments;
- mostly monochrome typography and thin rules;
- EHR-like field labels and timestamps;
- accessions/barcodes and signed/final status;
- scan artifacts on pathology/lab pages;
- different external-lab appearance for molecular testing;
- small synthetic label retained on every page.
