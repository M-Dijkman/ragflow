import pandas as pd

# Manually create lists of questions and answers based on your generated text
questions = [
    "What is the primary challenge in determining protein structures?",
    "What breakthrough capability does AlphaFold provide in protein structure prediction?",
    "What were the results of AlphaFold's performance in CASP14?",
    "What are the main components of AlphaFold’s neural network architecture?",
    "How does AlphaFold handle missing physical context in protein structures?",
    "What methods and data sources were used for AlphaFold’s training?",
    "What innovations in AlphaFold’s design contribute to its high accuracy?",
    "What are the limitations of AlphaFold identified in the article?",
    "How does AlphaFold measure the confidence in its predictions?",
    "What impact has AlphaFold demonstrated on the scientific community?"
]

answers = [
    "The primary challenge in determining protein structures is the bottleneck caused by the months to years of painstaking experimental effort required to determine a single protein structure. This has limited the structural coverage to around 100,000 unique proteins out of billions of known protein sequences.",
    "AlphaFold provides the first computational method capable of predicting protein structures with atomic accuracy in a majority of cases, even when no similar structure is known. This marks a significant advancement in addressing the protein folding problem.",
    "In CASP14, AlphaFold achieved a median backbone accuracy of 0.96 Å r.m.s.d.95 with a 95% confidence interval of 0.85–1.16 Å, significantly outperforming the next best method, which had a median accuracy of 2.8 Å r.m.s.d.95.",
    "AlphaFold’s neural network architecture includes two main stages: the trunk, which processes inputs through repeated Evoformer blocks to produce representations of the processed multiple sequence alignments (MSAs) and residue pairs, and the structure module, which refines the 3D protein structure with a focus on atomic details.",
    "AlphaFold is trained to produce protein structures most likely to appear in the Protein Data Bank (PDB) by implicitly respecting constraints such as stochiometry, ligand presence, or ion interactions. This allows it to handle underspecified structural conditions effectively.",
    "AlphaFold’s training utilized a combination of supervised learning on PDB data and a self-distillation approach. Input data included MSAs generated from databases such as UniRef90, BFD, and MGnify, and templates from PDB. The training emphasized evolutionary, physical, and geometric constraints.",
    "AlphaFold's high accuracy stems from innovations such as the Evoformer block for integrating evolutionary and pairwise relationships, the use of invariant point attention for geometry-aware updates, iterative refinement through recycling, and self-distillation for leveraging unlabelled sequence data.",
    "AlphaFold's accuracy decreases significantly when MSAs have a median alignment depth of fewer than 30 sequences. Additionally, it struggles with proteins having few intra-chain or homotypic contacts compared to heterotypic contacts, which are common in bridging domains within larger complexes.",
    "AlphaFold uses a predicted local-distance difference test (pLDDT) as a confidence measure, which correlates well with the Cα local-distance difference test (lDDT-Cα) accuracy of the predicted structure.",
    "AlphaFold has demonstrated significant utility in the experimental community, aiding in molecular replacement, interpreting cryogenic electron microscopy maps, and enabling proteome-scale predictions. It represents a step toward closing the gap between genomic sequencing and structural bioinformatics."
]

# Create the DataFrame
df = pd.DataFrame({
    "question": questions,
    "answer": answers
})

# Save to CSV
df.to_csv("./database/AlphaFold_qa.csv", index=False)