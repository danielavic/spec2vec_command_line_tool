import click
from datetime import datetime
import os
import io
from pyteomics import mgf
import requests
from matchms.importing import load_from_mgf
from data_processing import spectrum_processing
import gensim
from spec2vec import Spec2Vec
from matchms import calculate_scores
from graphml_and_edge_table import individual_graphml_and_edge_list, min_matched, from_pandas_edgelist, to_pandas_edgelist, scores_cosine
from download_trained_files import download_trained_models
import pandas as pd
import networkx as nx
import numpy as np
from matplotlib_venn import venn2
from matplotlib import pyplot as plt


@click.command()
@click.option("--taskid",
                  help="nothing", type=str)
@click.option("--trained_file",
                  help="nothing", type=str)
@click.option("--threshold_spec2vec",
                  help="nothing", type=float)
@click.option("--threshold_cosine",
                  help="nothing", type=float)
@click.option("--min_matches",
                  help="nothing", type=float)

def spectrum_similarity_tool(taskid, trained_file, threshold_spec2vec, threshold_cosine, min_matches):
    #Generate directory to save files
    current_datetime=datetime.now()
    dir_filename = f"{current_datetime}--{taskid}--threshold_spec2vec--{threshold_spec2vec}--threshold_cosine--{threshold_cosine}"
    dir_path=f"static/dir/{dir_filename}"
    os.mkdir(dir_path)
    #Generate and load spectra
    filename = f"{current_datetime}--{taskid}--threshold_spec2vec--{threshold_spec2vec}--threshold_cosine--{threshold_cosine}"
    path= f"static/mgf_files/{filename}"
    url_to_spectra = "http://gnps.ucsd.edu/ProteoSAFe/DownloadResultFile?task=%s&block=main&file=spec/" % taskid
    spectra = []
    with mgf.MGF(io.StringIO(requests.get(url_to_spectra).text)) as reader:
        for spectrum in reader:
            spectra.append(spectrum)

    mgf.write(spectra, output= path, key_order = ['feature_id', 'pepmass', 'scans', 'charge', 
                                                                    'mslevel', 'retention_time', 'precursor_mz',
                                                                   'ionmode', 'inchi', 'inchikey', 'smiles' ])
    
    spectra = list(load_from_mgf(path))

    #Processing spectra

    spectra = [spectrum_processing(s) for s in load_from_mgf(path)]
    spectra = [s for s in spectra if s is not None]

    # Download trained model files
    if trained_file == 'yes':
        model = gensim.models.Word2Vec.load("downloads/spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model")
    elif trained_file == 'no': 
        download_trained_models()
        path_model = os.path.join(os.path.dirname(os.getcwd()),
                    "data", "trained_models") 
                           # enter your path name when different
        filename_model = "downloads/spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model"
        filename = os.path.join(path_model, filename_model)
        model = gensim.models.Word2Vec.load("downloads/spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model")
    else: 
        print('Please, set the trained_file parameter to yes or no')

    # Calculate spec2vec similarity scores

    spec2vec_similarity = Spec2Vec(model=model, intensity_weighting_power=0.5, allowed_missing_percentage=5.0)
    scores_spec2vec = calculate_scores(spectra, spectra, spec2vec_similarity,
                          is_symmetric=True)
    individual_graphml_and_edge_list(scores_spec2vec, 'spec2vec', threshold_spec2vec)
    
    spec2vec_net = individual_graphml_and_edge_list.net

    spec2vec_net['source']=spec2vec_net['source'].astype(int)
    spec2vec_net['target']=spec2vec_net['target'].astype(int)

    G = from_pandas_edgelist(spec2vec_net, source='source', target='target', edge_attr=True)
    spec2vec_graphml = f"static/graphml_files/spec2vec/spec2vec--{current_datetime}--{taskid}.graphml"
    nx.write_graphml(G, spec2vec_graphml)

    print('calculei spec2vec')
    
    # Calculate cosine similarity scores

    cosine_net = scores_cosine(spectra, min_matches_filter=min_matches, scores=threshold_cosine)
    path_cosine_net_csv = f"static/csv_files/cosine--{current_datetime}--{taskid}--threshold_spec2vec--{threshold_spec2vec}--threshold_cosine--{threshold_cosine}.csv"
    cosine_net.to_csv(path_cosine_net_csv)
    cosine_net = pd.read_csv(path_cosine_net_csv)
    H = from_pandas_edgelist(cosine_net, source='source', target='target', edge_attr=True)
    cosine_graphml = f"static/graphml_files/cosine/cosine--{current_datetime}--{taskid}--threshold_spec2vec--{threshold_spec2vec}--threshold_cosine--{threshold_cosine}.graphml"
    nx.write_graphml(H, cosine_graphml)

    print('calculei cosseno')

    # Merge the DataFrames
    merged = pd.merge(spec2vec_net, cosine_net, on=['source', 'target'], how='outer', suffixes=('_spec2vec', '_cosine'))

    # Fill empty scores with zeros
    merged = merged.fillna(0)

    # Criando uma nova coluna indicando qual score está presente
    merged['score_type'] = np.where((merged['spec2vec'] > 0) & (merged['cosine'] > 0), 'both', 
                            np.where(merged['spec2vec'] > 0, 'spec2vec', 
                                    np.where(merged['cosine'] > 0, 'cosine', 'none')))
    
    filename_merged_csv_file = f"static/csv_files/merged--{current_datetime}--{taskid}.csv"

    
    # Saves as .csv
    merged.to_csv(filename_merged_csv_file, index=False)

    # Df for both scores
    merged = pd.read_csv(filename_merged_csv_file)
    
    # Create graph
    I = from_pandas_edgelist(merged, 'source', 'target', ['spec2vec', 'cosine', 'min_matches', 'score_type'])
    graphml_filename = f"static/graphml_files/both/both--{current_datetime}--{taskid}--threshold_spec2vec--{threshold_spec2vec}--threshold_cosine--{threshold_cosine}.graphml"

    # Saves graphml
    nx.write_graphml(I, graphml_filename)

    # Read df pandas
    df = pd.read_csv(filename_merged_csv_file)

    # Count relationships for each category
    spec2vec_only = df.score_type.value_counts().spec2vec
    cosine_only = df.score_type.value_counts().cosine
    both = df.score_type.value_counts().both

    # Venn Diagram
    venn2(subsets=(spec2vec_only, cosine_only, both), set_colors=('r', 'g', 'b'), set_labels=('Spec2Vec', 'Cosine', 'Both'))
    plt.savefig(f"static/venn_diagrams/venn_diagram--{current_datetime}--{taskid}.png")


if __name__ == "__main__":
    spectrum_similarity_tool()
