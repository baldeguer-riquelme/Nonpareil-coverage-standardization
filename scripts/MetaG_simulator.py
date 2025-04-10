#!/usr/bin/env python

# Authors: Borja Aldeguer-Riquelme
# Contact: briquelme3@gatech.edu

'''
This script builds in-silico short-read metagenomes with lognormal distributions. 
Several variables can be controlled, including the number of species, number of genomes per species, evenness, metagenome size and number of replicates. 
'''


import argparse
from glob import glob
import os
import random
import subprocess
import logging
import numpy as np
import matplotlib.pyplot as plt
import sys

def create_logger(out):
    # Open log file
    log_file =f"{out}.log" 

    # Remove log file if exists
    if os.path.isfile(log_file):
        os.remove(log_file)

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


def merge_fastq_files(input_files, output_file):
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
    content = glob(f"{dir}/*.fastq")
    for file in content:
        os.remove(file)
    #
    content = glob(f"{dir}/*.fai")
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
            genome_length = 0
            for line in file:
                if line.startswith(">"):
                    if seq_length > min_length:
                        output.write(''.join([header, seq, "\n"]))
                        genome_length += seq_length
                        #print('\t'.join([header, str(seq_length)]))
                    header = line
                    seq = ""
                    seq_length = 0
                else:
                    seq_length += len(line.replace("\n", ""))
                    seq += line.replace("\n", "")
            #
            if seq_length > min_length:
                output.write(''.join([header, seq, "\n"]))
                genome_length += seq_length
    #
    return(genome_length)


def simulate_illumina(genome, tmp_genome, read_length, num_sim_reads, threads, list_error_prob, out_file, prefix, log_file, logger):
    genome_length = fasta_filter(genome, tmp_genome, 2 * read_length)
    if list_error_prob:
        cmd = f"mason_simulator --ir {tmp_genome} \
            --illumina-read-length {read_length} --seq-technology illumina \
            -n {num_sim_reads} --num-threads {threads} \
            -o {out_file} --fragment-size-model normal\
            --read-name-prefix {prefix} \
            --illumina-prob-insert {list_error_prob[0]}\
            --illumina-prob-deletion {list_error_prob[1]}\
            --illumina-prob-mismatch {list_error_prob[2]}\
            --illumina-prob-mismatch-begin {list_error_prob[3]}\
            --illumina-prob-mismatch-end {list_error_prob[4]} >> {log_file} 2>&1"
    else:
        cmd = f"mason_simulator --ir {tmp_genome} \
            --illumina-read-length {read_length} --seq-technology illumina \
            -n {num_sim_reads} --num-threads {threads} \
            -o {out_file} --fragment-size-model normal\
            --read-name-prefix {prefix} >> {log_file} 2>&1"
    res_sim = subprocess.run(cmd, shell = True)
    os.remove(tmp_genome)
    if res_sim.returncode != 0:
        logger.error(f"Error! Short-read simulation failed with genome {genome}. Error code: {res_sim.returncode}")


def simulate_pacbio(genome, tmp_genome, read_length, read_length_sd, num_sim_reads, threads, out_file, out_path, prefix, log_file, logger):
    genome_length = fasta_filter(genome, tmp_genome, read_length)
    seq_depth = num_sim_reads * read_length / genome_length
    
    # Get .sam files containing simulated PacBio reads
    cmd = f"pbsim --genome {tmp_genome} --depth {seq_depth} \
    --prefix {out_path} --id-prefix {prefix} \
    --length-mean {read_length} --length-sd {read_length_sd} \
    --strategy wgs --method errhmm --errhmm ~/p-ktk3-0/anaconda3/data/ERRHMM-RSII.model --pass-num 10 >> {log_file} 2>&1"
    res_pbsim = subprocess.run(cmd, shell = True)
    if res_pbsim.returncode != 0:
        logger.error(f"Error! PacBio Long-read simulation failed with genome {genome}. Error code: {res_pbsim.returncode}")
    
    # Convert sam to bam and get HiFi reads from ccs (PacBio tool)
    list_sam = glob(f"{out_path}*.sam")
    for sam in list_sam:
        cmd = f"samtools view --threads {threads} -bS {sam} > {sam}.bam ; ccs --suppress-reports {sam}.bam {sam}.fastq"
        res_sam = subprocess.run(cmd, shell = True)
        if res_sam.returncode != 0:
            logger.error(f"Error! Samtools or ccs failed with genome {genome}. Error code: {res_sam.returncode}")
    
    # Merge multiple fastq into a single fastq file
    list_fastq = glob(f"{out_path}*.fastq")
    merge_fastq_files(list_fastq, out_file)

    # Remove tmp files
    tmp_to_rm = [file for x in [f"{out_path}*sam", f"{out_path}*bam", f"{out_path}*ref", f"{out_path}*maf", f"{out_path}*_report.txt", f"{out_path}*sam.fastq", f"{out_path}*metrics.json.gz"] for file in glob(x)]
    for file in tmp_to_rm:
        os.remove(file)


def simulate_reads(input, normalized_reads, num_species, num_genomes_per_sp, log_file, min_val, max_val, target_sum, read_length, read_length_sd, list_error_prob, ext, out, threads, logger, action):
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

        avail_genomes = glob(f"{folder}/*{ext}")
        selected_genomes = random.sample(avail_genomes, num_genomes_per_sp)
        genome_list.append(selected_genomes[0])

        genome_reads = get_points(min_val = min_val, max_val = max_val, num_points = num_genomes_per_sp, target_sum = target_sum, mu = 2, metag_size = sp_reads, plot = "False", out = out, logger = logger)

        # Generate short-reads for each selected genome
        for genome, num_sim_reads in zip(selected_genomes, genome_reads):
            
            prefix = genome.rsplit("/", 1)[1].rsplit(ext, 1)[0]
            out_path = f"tmp_{out}/{prefix}"
            tmp_genome = f"{out_path}_filter.fasta"
            
            logger.info(f"Simulating reads for genome {genome}")
            if action == "illumina":
                out_file = f"{out_path}_SE.fastq" # Name of the output file
                simulate_illumina(genome, tmp_genome, read_length, num_sim_reads, threads, list_error_prob, out_file, prefix, log_file, logger)
            elif action == "pacbio":
                out_file = f"{out_path}_PB.fastq" # Name of the output file
                simulate_pacbio(genome, tmp_genome, read_length, read_length_sd, num_sim_reads, threads, out_file, out_path, prefix, log_file, logger)
            sr_list.append(out_file)

    return(sr_list, genome_list)


def options(action):
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
    parser.add_argument("--t",
                        type=int,
                        help="Number of threads. Default: 1",
                        default = 1, 
                        required=False)
    #
    if action == "illumina":
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
        parser.add_argument("--error_prob",
            type=str,
            help="List of float values specifying read error probabilities in the following order: Probability insert, Probability deletion, Probability mismatch, Probability mismatch begin and Probability mismatch end. For error free reads use '0.0,0.0,0.0,0.0,0.0'. Default: mason_simulator default parameters",
            default = None, 
            required=False)
    #
    if action == "pacbio":
        parser.add_argument("--avg_read_length",
            type=int,
            help="Average length of simulated reads (in bases). Default: 9000",
            default = 9000, 
            required=False)
        parser.add_argument("--sd_read_length",
            type=int,
            help="Read length deviation of simulated reads (in bases). Default: 7000",
            default = 7000, 
            required=False)
        parser.add_argument("--metag_size",
            type=int,
            help="Number of reads per metagenome. Default: 100,000",
            default = 100000, 
            required=False)

    args, _ = parser.parse_known_args()
    return(args)


def main():
    accepted_actions = ["illumina", "pacbio"]

    if len(sys.argv) < 2 or sys.argv[1] in ["help", "--help", "-help", "-h"]:
        print("Please, specify one action of: 'illumina' or 'pacbio'")
        sys.exit()

    action = sys.argv[1]
    if action not in accepted_actions:
        print(f"{action} is not a recognised action. Please specify one action of: 'illumina' or 'pacbio'")

    args = options(action)

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
    threads = args.t # Number of threads to use
    out = args.out # Prefix for output files
    if action == "illumina":
        read_length = args.read_length
        read_length_sd = 30
        metag_size = args.metag_size
        error_prob = args.error_prob
        if error_prob:
            list_error_prob = [float(param) for param in error_prob.split(",")]
        else:
            list_error_prob = None
    elif action == "pacbio":
        read_length = args.avg_read_length
        read_length_sd = args.sd_read_length
        metag_size = args.metag_size
        list_error_prob = ""


    # 2. Open log file
    logger, log_file = create_logger(out)
    logger.info("Initializing metagenome simulation\n")


    # 3. Create tmp directory
    tmp_dir = f"tmp_{out}"
    if os.path.isdir(tmp_dir) == True:
        logger.info(f"Temporary directory {tmp_dir} already exists. Remove the folder or choose a different --out prefix")
        sys.exit()
    else:
        os.mkdir(tmp_dir)


    # 4. Build the simulated metagenome
    for rep in range(1, reps + 1):
        out_plot = f"{out}_metag_{rep}.pdf"
        sim_metag_file = f"{out}_metaG_{rep}.fastq"
        
        # 4.1. Get the number of reads per species
        normalized_reads = get_points(min_val = min_val, max_val = max_val, num_points = num_species, target_sum = target_sum, mu = mu, metag_size = metag_size, plot = "True", out = out_plot, logger = logger)

        # 4.2. Simulate reads for each species
        sr_list, genome_list = simulate_reads(input, normalized_reads, num_species, num_genomes_per_sp, log_file, min_val, max_val, target_sum, read_length, read_length_sd, list_error_prob, ext, out, threads, logger, action)

        # 4.3. Save list of selected genomes to file
        out_selected_genomes = f"{out}_{rep}_selected_genomes.txt"
        with open(out_selected_genomes, "w") as f:
            for line in genome_list:
                f.write(''.join([line, "\n"]))
            
        logger.info(f"Selected genomes saved to: {out_selected_genomes}")

        # 4.4. Combine simulated reads into one file
        merge_fastq_files(sr_list, sim_metag_file)
        logger.info(f"Simulated metagenome saved to: {sim_metag_file}\n")

        # 4.5. Clear tmp directory
        clear_directory(tmp_dir)
    

    os.rmdir(tmp_dir)



if __name__ == "__main__":
    main()