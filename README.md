# comp480-final-project-2019

Team members: Alexander Jiang, Jay Ryu

### Installation/Setup

    conda create -n newenv python=3.7
    conda activate newenv
    conda install -c conda-forge tqdm

### Run

To run the initial experiment (whether the dimensionality of objects inserted
into a Bloom filter affects the Bloom filter's false positive rate), run the
following command (the output figure will be saved to the "figures/" directory):

    python bloom_filter_dimensionality.py

To run the calculations of the false positive rate for our proposed model vs.
the standard Bloom filter, run the following command (the output will be printed
to the terminal; a copy of the output was saved in the "output" file):

    python calculate_fp_rate.py
