#!/usr/bin/env python3
"""
Evaluate log-likelihood using RAxML-NG with a GTR model equivalent to F81.

F81 model: All substitution rates are equal, only base frequencies vary.
In GTR parameterization: rates = 1/1/1/1/1/1 (all equal)
The mu parameter is set so that expected number of changes per site = 1:
    mu = 1 / (1 - sum(pi_i^2))

This script:
1. Reads the FASTA file to compute base composition (pi)
2. Creates a GTR model string with equal rates and computed frequencies
3. Runs RAxML-NG with --evaluate to compute log-likelihood without optimization
"""

import subprocess
import tempfile
import os
from pathlib import Path


def read_fasta(fasta_file):
    """Read FASTA file and return sequences as a dictionary."""
    sequences = {}
    current_name = None
    current_seq = []

    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_name is not None:
                    sequences[current_name] = ''.join(current_seq)
                current_name = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_name is not None:
            sequences[current_name] = ''.join(current_seq)

    return sequences


def calculate_base_composition(sequences):
    """Calculate base composition from sequences, ignoring gaps and ambiguous characters."""
    counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
    total = 0

    for seq in sequences.values():
        for base in seq.upper():
            if base in counts:
                counts[base] += 1
                total += 1

    if total == 0:
        raise ValueError("No valid nucleotides found in sequences")

    # Calculate frequencies in order A, C, G, T (RAxML-NG order)
    pi = {base: counts[base] / total for base in ['A', 'C', 'G', 'T']}
    return pi


def calculate_f81_mu(pi):
    """
    Calculate mu for F81 model so that expected rate = 1.
    mu = 1 / (1 - sum(pi_i^2))
    """
    sum_pi_squared = sum(p**2 for p in pi.values())
    denom = max(1e-14, 1.0 - sum_pi_squared)
    return 1.0 / denom


def create_gtr_model_string(pi, mu):
    """
    Create GTR model string for RAxML-NG equivalent to F81.

    For F81: all rates are equal, scaled by mu to normalize expected rate to 1.

    In GTR, the rate matrix Q[i][j] = r_ij * pi[j] for i != j
    The expected rate is sum_i sum_{j!=i} pi[i] * r_ij * pi[j]

    With all r_ij = 1, this gives sum_i pi[i] * (1 - pi[i]) = 1 - sum(pi^2)
    To normalize to 1, we need r_ij = mu = 1 / (1 - sum(pi^2))

    GTR rates order: AC, AG, AT, CG, CT, GT
    Frequencies order: A, C, G, T (fA, fC, fG, fT)

    Format: GTR{rAC/rAG/rAT/rCG/rCT/rGT}+FU{fA/fC/fG/fT}

    Using +FU (user-defined frequencies) to fix the frequencies.
    """
    # All rates equal to mu (scaled to normalize expected rate to 1)
    rates = f"{mu:.10f}/{mu:.10f}/{mu:.10f}/{mu:.10f}/{mu:.10f}/{mu:.10f}"

    # Frequencies in order A, C, G, T
    freqs = f"{pi['A']:.10f}/{pi['C']:.10f}/{pi['G']:.10f}/{pi['T']:.10f}"

    # Use GTR with fixed rates and user-defined frequencies
    # The +FU means user-defined fixed frequencies
    model_string = f"GTR{{{rates}}}+FU{{{freqs}}}"

    return model_string


def run_raxml_evaluate(fasta_file, tree_file, model_string, output_prefix="f81_eval"):
    """
    Run RAxML-NG in evaluate mode to compute log-likelihood.

    --evaluate: compute likelihood without optimization
    --model: the substitution model
    --msa: multiple sequence alignment
    --tree: tree topology with branch lengths
    --nofiles: minimize output files
    --opt-model off: do not optimize model parameters
    --opt-branches off: do not optimize branch lengths
    """

    cmd = [
        "raxml-ng",
        "--evaluate",
        "--msa", fasta_file,
        "--tree", tree_file,
        "--model", model_string,
        "--prefix", output_prefix,
        "--threads", "1",
        "--opt-model", "off",
        "--opt-branches", "off",
        "--redo"  # overwrite existing files
    ]

    print(f"Running command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("STDERR:", result.stderr)
        print("STDOUT:", result.stdout)
        raise RuntimeError(f"RAxML-NG failed with return code {result.returncode}")

    return result.stdout, result.stderr


def parse_raxml_output(output_prefix):
    """Parse RAxML-NG output to extract log-likelihood."""
    log_file = f"{output_prefix}.raxml.log"

    loglikelihood = None

    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                if "Final LogLikelihood:" in line:
                    loglikelihood = float(line.split()[-1])
                    break

    return loglikelihood


def main():
    # File paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"

    fasta_file = data_dir / "CDS_FcC_1000.fas"
    tree_file = data_dir / "RAxML_bipartitions.CDS_FcC_partition"

    # Check files exist
    if not fasta_file.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_file}")
    if not tree_file.exists():
        raise FileNotFoundError(f"Tree file not found: {tree_file}")

    print(f"Reading sequences from: {fasta_file}")
    sequences = read_fasta(str(fasta_file))
    print(f"Found {len(sequences)} sequences")

    # Calculate base composition
    pi = calculate_base_composition(sequences)
    print(f"\nBase composition (pi):")
    for base in ['A', 'C', 'G', 'T']:
        print(f"  {base}: {pi[base]:.10f}")

    # Calculate F81 mu
    mu = calculate_f81_mu(pi)
    print(f"\nF81 mu (for expected rate = 1): {mu:.10f}")

    # Verify: expected rate should be 1
    sum_pi_squared = sum(p**2 for p in pi.values())
    expected_rate = mu * (1.0 - sum_pi_squared)
    print(f"Verification - Expected rate: {expected_rate:.10f} (should be 1.0)")

    # Create GTR model string with mu-scaled rates
    model_string = create_gtr_model_string(pi, mu)
    print(f"\nGTR model string (equivalent to F81):")
    print(f"  {model_string}")

    # Run RAxML-NG
    output_prefix = str(script_dir / "f81_eval")
    print(f"\nRunning RAxML-NG evaluation...")

    try:
        stdout, stderr = run_raxml_evaluate(
            str(fasta_file),
            str(tree_file),
            model_string,
            output_prefix
        )

        # Parse results
        loglikelihood = parse_raxml_output(output_prefix)

        if loglikelihood is not None:
            print(f"\n{'='*50}")
            print(f"Log-Likelihood: {loglikelihood:.6f}")
            print(f"{'='*50}")
        else:
            print("\nCould not parse log-likelihood from output")
            print("Check the log file:", f"{output_prefix}.raxml.log")

        # Also print any relevant info from stdout
        if stdout:
            print("\nRAxML-NG output:")
            for line in stdout.split('\n'):
                if 'likelihood' in line.lower() or 'error' in line.lower():
                    print(f"  {line}")

    except Exception as e:
        print(f"Error running RAxML-NG: {e}")
        raise


if __name__ == "__main__":
    main()
