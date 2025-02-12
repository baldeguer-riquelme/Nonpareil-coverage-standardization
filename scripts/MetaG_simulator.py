#!/usr/bin/env python

# Authors: Borja Aldeguer-Riquelme
# Contact: briquelme3@gatech.edu

'''
This script builds in-silico short-read metagenomes with lognormal distributions. 
Several variables can be controlled, including the number of species, number of genomes per species, evenness, metagenome size and number of replicates. 
'''


import argparse
import glob
import os
import random
import subprocess
import logging
import numpy as np
import matplotlib.pyplot as plt

def create_logger(out):
    # Open log file
    log_file =f"{out}.log" 

    # Create a logger
    logger = logging.getLogger('logs')
    logger.setLevel(logging.DEBUG)

    # Create a formatter to define the log format
    formatter = logging.Formatter("%(asctime)s :: %(levelname)s :: %(message)s", datefmt = "%d-%m-%Y %H:%M:%S")

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(filename = log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Create a stream handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return(logger, log_file)


def merge_fastq_files(input_files, output_file, logger):
    """
    Merges multiple FASTQ files into a single file.

    Parameters:
        input_files (list of str): List of input FASTQ file paths.
        output_file (str): Output FASTQ file path.
    """

    buffer_size = 4*1024*1024
    with open(output_file, 'wb') as outfile:
        for file in input_files:
            with open(file, 'rb') as infile:
                while chunk := infile.read(buffer_size):
                    outfile.write(chunk)
    
    #logger.info(f"Merged FASTQ written to {output_file}\n")


def clear_directory(dir):
    """
    Removes files in a given directory

    Parameters:
        dir (str): Directory with files to remove
    """
    content = glob.glob(f"{dir}/*.fastq")
    for file in content:
        os.remove(file)


def get_points(min_val, max_val, num_points, target_sum, mu, metag_size, plot, out, logger):
    # Generate a log-normal distribution
    mu = mu  # Use 2 for uneven and 5 for more even distribution
    sigma = 0.8  # Standard deviation of the underlying normal distribution
    sample_size = int(1e6)  # Large sample size for smooth filtering
    lognorm_data = np.random.lognormal(mean=mu, sigma=sigma, size=sample_size)

    # Filter and sort data within range
    filtered_data = lognorm_data[(lognorm_data >= min_val) & (lognorm_data <= max_val)]
    sorted_data = np.sort(filtered_data)

    # Select evenly spaced points
    evenly_spaced_indices = np.linspace(0, len(sorted_data) - 1, num_points, dtype=int)
    selected_points = sorted_data[evenly_spaced_indices]

    # Normalize the points so their sum equals the target sum
    normalized_points = selected_points / np.sum(selected_points) * target_sum
    normalized_reads = (normalized_points * metag_size).astype(int)

    # Validate constraints
    assert np.isclose(np.sum(normalized_points), target_sum), f"Normalization failed, sum = {np.sum(normalized_points)}"

    if plot == "True":
        # Plot for visualization
        #plt.figure(figsize=(8, 6))
        plt.plot(normalized_reads, 'o-', color='blue', label='Normalized Points')
        plt.title('Log-Normal Distribution', fontsize=14)
        plt.xlabel('Genome', fontsize=12)
        plt.ylabel('Reads', fontsize=12)
        plt.savefig(out, format="pdf", bbox_inches="tight")
        logger.info(f"Saving log-normalized distribution plot to {out}")
        #plt.show()

    return(normalized_reads)


def fasta_filter(input, tmp_input, min_length):
    with open(input, "r") as file:
        with open(tmp_input, "w") as output:
            header = ""
            seq_length = 0
            seq = ""
            for line in file:
                if line.startswith(">"):
                    if seq_length > min_length:
                        output.write(''.join([header, seq, "\n"]))
                        #output.write(seq)
                        print('\t'.join([header, str(seq_length)]))
                    header = line
                    seq = ""
                    seq_length = 0
                else:
                    seq_length += len(line.replace("\n", ""))
                    seq += line.replace("\n", "")
    #
            if seq_length > min_length:
                        output.write(''.join([header, seq, "\n"]))


def simulate_reads(input, normalized_reads, num_species, num_genomes_per_sp, log_file, min_val, max_val, target_sum, read_length, ext, out, threads, logger):
    # Read input list of folders
    with open(input, "r") as f:
        folder_list = f.readlines()
        folder_list = [line.rstrip('\n') for line in folder_list]


    # Randomly select genomes for each species
    genome_list = []
    sr_list = []
    for folder, sp_reads in zip(folder_list[0:num_species], normalized_reads): # Get only X number of species
        folder_split = folder.rsplit("/")
        sp = [folder_split[-1] if len(folder_split[-1]) > 1 else folder_split[-2]][0]
        logger.info(f"Processing species {sp}")

        avail_genomes = glob.glob(f"{folder}/*{ext}")
        selected_genomes = random.sample(avail_genomes, num_genomes_per_sp)
        genome_list.append(selected_genomes[0])

        genome_reads = get_points(min_val = min_val, max_val = max_val, num_points = num_genomes_per_sp, target_sum = target_sum, mu = 2, metag_size = sp_reads, plot = "False", out = out, logger = logger)

        # Generate short-reads for each selected genome
        for genome, num_sim_reads in zip(selected_genomes, genome_reads):
            
            prefix = genome.rsplit("/", 1)[1].rsplit(ext, 1)[0]
            out_path = f"tmp_{out}/{prefix}"
            tmp_genome = f"{out_path}_filter.fasta"
            out_file = f"{out_path}_SE.fastq" # Name of the output file
            sr_list.append(out_file)
            
            logger.info(f"Simulating reads for genome {genome}")
            fasta_filter(genome, tmp_genome, 2 * read_length)
            cmd = f"mason_simulator --ir {tmp_genome} \
                --illumina-read-length {read_length} --seq-technology illumina \
                -n {num_sim_reads} --num-threads {threads} \
                -o {out_file} \
                --read-name-prefix {prefix} >> {log_file} 2>&1"
            res_sim = subprocess.run(cmd, shell = True)
            os.remove(tmp_genome)
            if res_sim.returncode != 0:
                logger.error(f"Error! Short-read simulation failed with genome {genome}. Error code: {res_sim.returncode}")


    return(sr_list, genome_list)


def main():
    parser = argparse.ArgumentParser("Generates in-silico metagenomes with user defined characteristics (e.g., number of species, metagenome size, eveness, number of genomes per species)")
    parser.add_argument("--genome", 
                        type=str, 
                        help="File containing the path to the folder containing the genomes of each species (one per line)", 
                        required=True)
    parser.add_argument("--out",
                        type=str,
                        help="Prefix of output files. Required",
                        required=True)
    parser.add_argument("--num_sp",
                        type=int,
                        help="Number of species to include in the metagenome. Default: 100",
                        default = 100, 
                        required=False)
    parser.add_argument("--num_genomes_per_sp",
                        type=int,
                        help="Number of genomes per species. Default: 1", 
                        default = 1,
                        required=False)
    parser.add_argument("--extension",
                        type=str,
                        help="Extension of genome fasta files. Default: .fna",
                        default = ".fna", 
                        required=False)
    parser.add_argument("--max_value",
                        type=float,
                        help="The ratio between --max_value and --min_value determines the difference between the most and less abundant species. For example, --max_value 1000 --min_value 0.1 indicates that the most abundant species will be 10,000 times more abundant than the less abundant one. It allows to control the evenness of the metagenome. Default: 1000",
                        default = 1000, 
                        required=False)
    parser.add_argument("--min_value",
                        type=float,
                        help="The ratio between --max_value and --min_value determines the difference between the most and less abundant species. For example, --max_value 1000 --min_value 0.1 indicates that the most abundant species will be 10,000 times more abundant than the less abundant one. It allows to control the evenness of the metagenome. Default: 0.1",
                        default = 0.1, 
                        required=False)
    parser.add_argument("--mu",
                        type=int,
                        help="Mu factor of the log-normal distribution. It allows to control the evenness of the metagenome. Use 2 for uneven and 5 for more even distribution. Default: 2",
                        default = 2, 
                        required=False)
    parser.add_argument("--num_metagenomes",
                        type=int,
                        help="Number of metagenomes to generate. Default: 3",
                        default = 3, 
                        required=False)
    parser.add_argument("--read_length",
                        type=int,
                        help="Length of simulated reads (in bases). Default: 150",
                        default = 150, 
                        required=False)
    parser.add_argument("--metag_size",
                        type=int,
                        help="Number of reads per metagenome. Default: 30,000,000",
                        default = 30000000, 
                        required=False)
    parser.add_argument("--t",
                        type=int,
                        help="Number of threads. Default: 1",
                        default = 1, 
                        required=False)
    args = parser.parse_args()


    # 1. Define parameters
    # Default
    target_sum = 1.0  # Aggregated sum of returned points

    # User defined
    input = args.genome
    num_species = args.num_sp  # Number of species
    num_genomes_per_sp = args.num_genomes_per_sp # Number of genomes per species
    ext = args.extension # Extension of genomes
    mu = args.mu # Mu factor of the log-normal distribution
    min_val = args.min_value  # Minimum value
    max_val = args.max_value   # Maximum value
    reps = args.num_metagenomes # Total number of metagenomes
    read_length = args.read_length # Read length
    metag_size = args.metag_size # Metagenome size (total number of reads)
    threads = args.t # Number of threads to use
    out = args.out # Prefix for output files


    # 2. Open log file
    logger, log_file = create_logger(out)
    logger.info("Initializing metagenome simulation\n")


    # 3. Create tmp directory
    tmp_dir = f"tmp_{out}"
    if os.path.isdir(tmp_dir) == True:
        logger.info(f"Temporary directory {tmp_dir} already exists. Remove the folder or choose a different --out prefix")
    else:
        os.mkdir(tmp_dir)


    # 4. Build the simulated metagenome
    for rep in range(1, reps + 1):
        out_plot = f"{out}_metag_{rep}.pdf"
        sim_metag_file = f"{out}_metaG_{rep}.fastq"
        
        # 4.1. Get the number of reads per species
        normalized_reads = get_points(min_val = min_val, max_val = max_val, num_points = num_species, target_sum = target_sum, mu = mu, metag_size = metag_size, plot = "True", out = out_plot, logger = logger)

        # 4.2. Simulate reads for each species
        sr_list, genome_list = simulate_reads(input, normalized_reads, num_species, num_genomes_per_sp, log_file, min_val, max_val, target_sum, read_length, ext, out, threads, logger)

        # 4.3. Save list of selected genomes to file
        out_selected_genomes = f"{out}_{rep}_selected_genomes.txt"
        with open(out_selected_genomes, "w") as f:
            for line in genome_list:
                f.write(''.join([line, "\n"]))
            
        logger.info(f"Selected genomes saved to: {out_selected_genomes}")

        # 4.4. Combine simulated reads into one file
        merge_fastq_files(sr_list, sim_metag_file, logger)
        logger.info(f"Simulated metagenome saved to: {sim_metag_file}\n")

        # 4.5. Clear tmp directory
        clear_directory(tmp_dir)
    

    os.rmdir(tmp_dir)



if __name__ == "__main__":
    main()