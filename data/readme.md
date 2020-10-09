##### Use case: Variant Effect prediction with machine Learnign.
    * This first notebook is intended for setting up our dataset.
    * I had downloaded (17-Jun-2020) and annotated the whole clinvar Dataset. 
 
##### Some comments
* Trabajamos sólo con variantes sin conflicto de interpretación que están anotadas en alguna de las siguientes categorías: 
    * Benign | Likely Bening | Beningn/Likely Beningn
    * Pathogenic| Likely Pathogenic| Pathogenic/Likely Pathogenic
* Descartamos variantes con inconsistencias en los campos ClinicalSignificance - ClinSigSimple 
* Descartamos variantes con errores en la anotación de alguno de los alelos


##### Por simplicidad, el dataset fue annotado mediante OpenCravat. 
* Incluimos features de frecuencia poblacional, de evolución, de redes de interacción, de significancia clínica, predictores, etc. 


#### Mantenemos sólo las variantes Missense, 
* (otras variantes están subrepresentadas las variantes benignas) i.e. stop lost, splicing. 

## Descripción de variables

***Essential Essential or non-essential genes***
CRISPR Essential ("E") or Non-essential phenotype-changing ("N") based on large scale CRISPR experiments.
CRISPR2 Essential ("E"), context-Specific essential ("S"), or Non-essential phenotype-changing ("N") based on large scale CRISPR experiments.
Gene Trap Essential ("E"), HAP1-Specific essential ("H"), KBM7-Specific essential ("K"), or Non-essential phenotype-changing ("N"), based on large scale mutagenesis experiments.


***Indispensability Score*** A probability prediction of the gene being essential.
Indispensability Prediction based on Gene_indispensability_score.
Essential ("E") or loss-of-function tolerant ("N")


***GHIS***
GHIS is a database providing haploinsufficiency scores derived from a combination of unbiased large-scale high-throughput datasets, including gene co-expression and genetic variation in over 6000 human exomes. These scores can readily be used to prioritize gene disruptions resulting from any genetic variant, including copy number variants, indels and single-nucleotide variants

***RVIS***
RVIS is a database providing variation intolerance scoring that assesses whether genes have relatively more or less functional genetic variation than expected based on the apparently neutral variation found in the gene. Scores were developed using sequence data from 6503 whole exome sequences made available by the NHLBI Exome Sequencing Project (ESP).

Score A measure of intolerance of mutational burden
Percentile Rank The percentile rank of the gene based on RVIS

***ExAC based features***
FDR p-value A gene's FDR p-value for preferential LoF depletion among ExAC
ExAC-based RVIS ExAC-based RVIS, where 'common' MAF is 0.05% in at least one population
ExAC-based Percentile Genome-Wide percentile for the ExAC-based RVIS

***Gnomad Gene features***
Obv/Exp LoF Observed/Expected for loss of function variants
Obv/Exp Mis Observed/Expected for missense variants
Obv/Exp Syn Observed/Expected for synonymous variants
LoF Z-Score Z-score for loss of function variants
Mis Z-Score Z-score for missense variants
Syn Z-Score Z-score for synonymous variants
pLI Probability of being loss-of-function intolerant
pRec Probability of being tolerant of homozygous LOF variants
pNull Probability of being tolerant of heterozygous and homozygous LOF variants

***PHI & PREC***
P(HI) Estimated probability of haploinsufficiency of the gene
P(rec) Estimated probability that gene is a recessive disease gene
Known Status Known recessive status of the gene. lof-tolerant: seen in homozygous state in at least one 1000G individual. recessive: known OMIM recessive disease.

