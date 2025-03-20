# Nonpareil coverage standardization
Repository with scripts to standardize relative abundance (TAD/GEQ, RPKM or recruited reads) to a given Nonpareil coverage (Npc).

### Install dependencies
[tidyverse](https://www.tidyverse.org)

[reshape2](https://cran.r-project.org/web/packages/reshape2/index.html)

[nonpareil](https://cran.r-project.org/web/packages/Nonpareil/index.html)

[roxygen2](https://roxygen2.r-lib.org/index.html)

[bbmap](https://jgi.doe.gov/data-and-tools/software-tools/bbtools/bb-tools-user-guide/bbmap-guide/)


```
conda create -n npc_stand -c conda-forge -c bioconda r-tidyverse r-reshape2 nonpareil r-roxygen2 bbmap
```
```
conda activate npc_stand
```

To install argparser:
```
# In terminal
git clone https://bitbucket.org/djhshih/argparser.git
cd argparser
R
```
```
# In R
library(roxygen2)
roxygenize()
quit()
```
```
# In terminal
R CMD INSTALL .
```
### Clone the repository
```
git clone https://github.com/baldeguer-riquelme/Nonpareil-coverage-standardization.git
```
### Test installation
To verify that the script runs without errors, run a test with the files placed in the "examples" folder.

```
cp Nonpareil-coverage-standardization/scripts/Npc_normalizer.R Nonpareil-coverage-standardization/examples
cd Nonpareil-coverage-standardization/examples
Rscript Npc_standardization.R -a Matrix_TAD_GEQ.txt -f Feature_metadata.txt -m Metagenome_metadata.txt -i TAD_GEQ -n 0.5 
```

The output should look like:

```
── Attaching core tidyverse packages ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── tidyverse 2.0.0 ──
✔ dplyr     1.1.4     ✔ readr     2.1.5
✔ forcats   1.0.0     ✔ stringr   1.5.1
✔ ggplot2   3.5.1     ✔ tibble    3.2.1
✔ lubridate 1.9.3     ✔ tidyr     1.3.1
✔ purrr     1.0.2
── Conflicts ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── tidyverse_conflicts() ──
✖ dplyr::filter() masks stats::filter()
✖ dplyr::lag()    masks stats::lag()
ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors

Attaching package: ‘reshape2’

The following object is masked from ‘package:tidyr’:

    smiths


[ 2024-06-18 16:22:05 ] Reading files...
Using Gene as id variables

[ 2024-06-18 16:22:05 ] Calculating nonpareil coverage from .npo files...

[ 2024-06-18 16:22:14 ] Using nonpareil coverage value provided by user (0.5) to standardize relative abundance...

[ 2024-06-18 16:22:15 ] Standardizing data...

[ 2024-06-18 16:22:47 ] Saving standardization matrix to file: Standardized_abundance_Npc_0.5.txt
```

## How to run
## 1. Check if your data requires Nonpareil coverage standardization based on ΔNpc_max
In the case that the Nonpareil coverage of your metagenome(s) is different between the factors you are interested in (e.g., treated vs control metagenomes) you should calculate ΔNpc_max to assess whether such difference could have an impact on your results. You can also calculate this metric to determine the minimum Npc needed to detect a genome/gene in one metagenome. The script Npc_max.R was developed for this purpose and you will need:

1. Abundance matrix. Should be tab delimited matrix with features (MAGs, genes) in rows and samples in columns. Only TAD/GEQ is accepted for now. We plan to incorporate RPKM in the future.
2. MAG metadata. MAG IDs and length (bp) should be provided in 2 columns named as "MAG" and "Length". Genes can be used instead of MAGs but the column header should be kept as "MAG". Feature factors (e.g., taxonomy or gene function) along with their corresponding levels (e.g., ) can be added in the third and following columns.
3. Metagenome metadata. Metagenome IDs and genome equivalents (GEQ) should be provided with at least 2 columns named as "Metagenome" and "GEQ". Metagenome factors (e.g., Antibiotic treatment) along with their corresponding levels (e.g., treated vs untreated) can be added in the third and following columns.
4. .npo files generated after running nonpareil (stand-alone) with each of your metagenomes of interest. These files should be placed on the same folder where you are running the analysis. For instructions about how to run nonpareil visit https://nonpareil.readthedocs.io/en/latest/index.html

Once you have them, you are ready to calculate ΔNpc_max running:
```
Rscript Npc_max.R -a Matrix_TAD_GEQ.txt -f Feature_metadata.txt -m Metagenome_metadata.txt --npo npo_folder/ -o output_test
```

Different statistical approaches will be used depending on the number of features and metagenomes per factor level:

|              | 1 metagenome                                        | ≥ 2 metagenomes                                         |
|--------------|-----------------------------------------------------|---------------------------------------------------------|
| 1 feature    | Minimum Npc to detect the feature in the metagenome | t.test of the aggregated relative abundances per sample |
| ≥ 2 features | t.test                                              | t.test of the aggregated relative abundances per sample |

The output will be 1) a pdf with boxplots displaying the abundance and number of features detected at each Npc and 2) a tab delimited table with the ΔNpc_max for each metagenome and feature factor. In the case that factors are not provided, the script will calculate the minimum Npc needed to detected each individual feature (genome or gene) in each metagenome. If ΔNpc_max of your feature(s) of interest is lower than the actual Npc difference between your metagenomes you will need to normalize your metagenomes to the same Npc (see section 2 below).

## 2. Standardization
In the case that your data requires Nonpareil coverage standardization, there are two options: 1) subsampling the metagenome or 2) estimate relative abundances.

### 2.1. Metagenome subsampling
To get your subsampled metagenome at a given Nonpareil coverage you will need:

1. Metagenome in FastQ/A format. Single or pair end.
2. npo files from Nonpareil analysis 

Then, you will have to choose the Nonpareil coverage value you would like to subsample your metagenome and run:
```
# For single end metagenomes
Rscript Npc_standardization_manual.R --m metagenome.fastq --npo metagenome.npo --cov 0.6
```

```
# For pair end metagenomes
Rscript Npc_standardization_manual.R --m metagenome_1.fastq --m2 metagenome_2.fastq --npo metagenome.npo --cov 0.6
```

### 2.2. Estimate relative abundances
To standardize your data, you will need:

1. Abundance matrix. Should be tab delimited matrix with features (MAGs, genes) in rows and samples in columns. Accepted abundance metrics are: TAD/GEQ, RPKM or reads.
2. MAG metadata. MAG IDs and length (bp) should be provided in 2 columns named as "MAG" and "Length". Genes can be used instead of MAGs but the column header should be kept as "MAG".
3. Metagenome metadata. Metagenome IDs and genome equivalents (GEQ) should be provided in 2 columns named as "Metagenome" and "GEQ". This is only required if data is TAD/GEQ, ignore it for RPKM or reads.
4. .npo files generated after running nonpareil (stand-alone) with each of your metagenomes of interest. These files should be placed on the same folder where you are running the analysis. For instructions about how to run nonpareil visit https://nonpareil.readthedocs.io/en/latest/index.html



Once you have prepared your tables, you are ready to standardize your data in a matter of seconds!


For TAD/GEQ data:
```
Rscript Npc_standardization.R -a Matrix_TAD_GEQ.txt -f Feature_metadata.txt -m Metagenome_metadata.txt -i TAD_GEQ -n 0.6 
```

For RPKM data:
```
Rscript Npc_standardization.R -a Matrix_RPKM.txt -f Feature_metadata.txt -i RPKM -n 0.6
```

For reads counts:
```
Rscript Npc_standardization.R -a Matrix_Reads.txt -f Feature_metadata.txt -i reads -n 0.6
```

## 3. Build *in-silico* metagenomes with user-defined characteristics
In-silico metagenomes are a great tool for assessing the impact of Nonpareil coverage as well as for many other purposes. To facilitate this task and enhance reproducibility, we implemented this process in the MetaG_simulator.py script which allows controlling several metagenome characteristics such as number of species, number of genomes per species (microdiversity), evenness, metagenome size and number of replicates. The genome distribution both at the interspecies and intraspecies level, is fitted to a lognormal distribution which can be adapted to the users's interests by tunning several parameters. 

### Dependencies
[Mason](https://www.seqan.de/apps/mason.html)

[NumPy](https://numpy.org/citing-numpy/)

[Matplotlib](https://matplotlib.org/stable/project/citing.html)

```
conda install -c conda-forge -c bioconda numpy matplotlib mason
```

### Run
The script only requires the following inputs to run:

1. File containing the path to the folder containing the genomes of each species (one per line). Genomes will be processed in the same order they are in the file. 
2. Output prefix.

Then, you can run the script using default parameters to simulate Illumina or PacBio metagenomes:
```
python MetaG_simulator.py illumina --genome genome_paths.txt --out test
```
```
python MetaG_simulator.py pacbio --genome genome_paths.txt --out test
```

By default, the metagenome will have an uneven genome distribution with a few abundant species and many low abundant species. However, you can modify this by tunning --max_value , --min_value and --mu. The ratio max_value / min_value determines the difference between the highest and the lowest abundant species. For example, if max_value = 100 and min_value = 1, the most abundant species will be 100 times more abundant than the less abundant one. By default, the ratio is 10,000. The --mu parameter controls for the mean of the underlying normal distribution (for further details, see the "mean" argument of the numpy.random.lognormal function [https://numpy.org/doc/2.1/reference/random/generated/numpy.random.lognormal.html]). The default mu value is 2 which provides an uneven metagenome but could be increased to 5, for example, to make the distribution more even.
```
python MetaG_simulator.py illumina --genome genome_paths.txt --out test --max_value 500 --min_value 1 --mu 5
```

The output will include:
1. the simulated fastq metagenome;

2. a pdf with the plot of the lognormal distribution applied;

3. the list of genomes included in the metagenome;

4. a log file. Check it if facing any issues.


