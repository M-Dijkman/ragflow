import pickle

import cuml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystow
import seaborn as sns
from textblob import TextBlob

DATA_DIR = pystow.join("AssayCTX", "data")

# Use TextBlob
def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words


def UMAP_main_text():
    columns = ['assay_type', 'assay_tax_id', 'confidence_score', 'curated_by', 'standard_type', 'bao_format']
    names = {'assay_type': 'Assay type', 'assay_tax_id': 'Taxonomy', 'confidence_score': 'Confidence score', 'curated_by': 'Curation'
             , 'standard_type': 'Standard type', 'bao_format': 'Format'}

    addition[columns] = helper[columns]

    plt.figure(figsize=(10, 9), dpi=600)
    plt.subplots_adjust(hspace=0.5)


    for i, color_by in enumerate(columns):
        ax = plt.subplot(2, 3, i + 1)
        # bg points
        sns.scatterplot(
            data=addition,
            x="x",
            y="y",
            c="grey",
            alpha=0.1,
            s=4,
            ax=ax,
        )

        addition[color_by] = addition[color_by].astype(str)
        if len(addition[color_by].unique()) > 5:
            plot_df = addition.loc[addition[color_by].isin(addition[color_by].value_counts()[:5].index.tolist())]
        else:
            plot_df = addition
        plot_df[color_by] = plot_df[color_by].apply(lambda x : x.replace('assay_tax_id_', '').replace('confidence_score_', '').replace('_', ' ').replace('.0', '').replace('BAO ', ''))
        # if i > 3: continue
        
        # ax.scatter(plot_df['x'], plot_df['y'], s=4, c=plot_df[color_by], cmap='bright')
        # labeled points
        sns.scatterplot(
            data=plot_df,
            x="x",
            y="y",
            hue=color_by,
            hue_order=sorted(plot_df[color_by].unique().tolist()),
            palette="bright",
            s=4,
            ax=ax,
        )
        if descriptor == 'BOW':
            lim = 8
        else:
            lim = 6
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(names[color_by], fontsize=14, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set(xticklabels=[], yticklabels=[])
        if color_by == 'assay_type':
            labels = ['Binding',  'Functional']
        elif color_by == 'bao_format':
            labels = ['Assay', 'Organism-based', 'Cell-based', 'Tissue-based', 'Single protein']
        elif color_by == 'assay_tax_id':
            labels = ['M. musculus', 'R. norvegicus', 'S. aureus', 'H. sapiens', 'Not defined']
        else:
            labels = None
        ax.legend(fontsize=8, handletextpad=0, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, bbox_transform=ax.transAxes)
        for count, legend_handle in enumerate(ax.get_legend().legend_handles):
            legend_handle.set(markersize = 5, alpha = 0.8, markeredgewidth=0)
        if labels:
            leg =ax.get_legend()
            for t, label in zip(leg.texts, labels):
                t.set_text(label)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{descriptor}_main_text.png")


def UMAP_topics():
    embeddings = np.array(helper["word_vectors"].to_list())
    embedding = reducer.fit_transform(embeddings)

    addition = pd.DataFrame(embedding, columns=["x", "y", "z1", "z2", "z3"])

    columns = ['cluster_None', 'olr_cluster_None']

    addition[columns] = helper[columns]

    plt.figure(figsize=(6, 4), dpi=600)
    plt.subplots_adjust(hspace=0.5)


    for i, color_by in enumerate(columns):
        ax = plt.subplot(1, 2, i + 1)
        # bg points
        sns.scatterplot(
            data=addition,
            x="x",
            y="y",
            c="grey",
            alpha=0.1,
            s=4,
            ax=ax,
        )

        addition[color_by] = addition[color_by].astype(str)
        counts = addition[color_by].value_counts()
        counts = counts.drop('-1')
        plot_df = addition.loc[addition[color_by].isin(counts[:10].index.tolist())]
        
        plot_df[color_by] = plot_df[color_by].apply(lambda x : x.replace('assay_tax_id_', '').replace('confidence_score_', '').replace('_', ' ').replace('.0', '').replace('BAO ', ''))
        # if i > 3: continue
        
        # ax.scatter(plot_df['x'], plot_df['y'], s=4, c=plot_df[color_by], cmap='bright')
        # labeled points
        sns.scatterplot(
            data=plot_df,
            x="x",
            y="y",
            hue=color_by,
            hue_order=sorted(plot_df[color_by].unique().tolist()),
            palette="bright",
            s=4,
            ax=ax,
            legend=False
        )
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_title(f'{"Without outlier reduction" if i == 0 else "With outlier reduction"}', fontsize=14, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set(xticklabels=[], yticklabels=[])
        
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(FIG_DIR / "embeddings_topics.png")


def UMAP_SI():
    columns = ['assay_type', 'topic',
        'assay_test_type', 'assay_category', 'assay_cell_type',
        'assay_organism', 'assay_tax_id', 'assay_strain', 'assay_tissue',
        'assay_subcellular_fraction', 'relationship_type', 'bao_format',
        'confidence_score', 'curated_by', 'src_id', 'cell_id',
        'tissue_id', 'variant_id', 'aidx', 'pref_name', 'standard_type',
        'journal', 'year', 'pubmed_id', 'doi']

    columns = [column for column in columns if column in helper.columns.to_list()]
    # add helper["topic"] to addition
    addition[columns] = helper[columns]

    plt.figure(figsize=(10, 10), dpi=600)
    plt.subplots_adjust(hspace=0.5)

    for i, color_by in enumerate(columns):
        addition[color_by] = addition[color_by].astype(str)
        if len(addition[color_by].unique()) > 5:
            plot_df = addition.loc[addition[color_by].isin(addition[color_by].value_counts()[:5].index.tolist())]
        else:
            plot_df = addition
        # if i > 3: continue
        ax = plt.subplot(5, 5, i + 1)
        sns.scatterplot(
            # data=dataframe,
            x=plot_df["x"],
            y=plot_df["y"],
            hue=plot_df[color_by],
            palette="bright",
            s=4,
            ax=ax,
            legend=True,
        )
        if descriptor == 'BOW':
            lim = 8
        else:
            lim = 6
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"{color_by.replace('_x', '').replace('_', ' ').title()}", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set(xticklabels=[], yticklabels=[])
        labels = sorted([string.replace('_', ' ') for string in plot_df[color_by].unique().tolist()])
        ax.legend(fontsize=3, loc='upper left', labels=labels)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{descriptor}_embeddings_SI.png")

if __name__ == "__main__":
    # set data directories using pystow
    DATA_DIR = pystow.join("AssayCTX", "data")

    # set figure directory using pystow
    FIG_DIR = pystow.join("AssayCTX", "figures")

    info = pd.read_csv(DATA_DIR / "assay_desc_mapping_fb_info.csv")

    descriptor = 'BioBERT'
    if descriptor == 'BOW':
        word_vector = 'word_vectors_BOW'

        with open(DATA_DIR / 'chembl_vectorizer.pk', 'rb') as fn:
            vectorizer = pickle.load(fn)

        info = info.sample(frac=0.1, random_state=40)

        corpus = info.description
        vectors = vectorizer.transform(corpus)
        
        info[word_vector] = vectors.toarray().tolist()
    elif descriptor == 'BioBERT':
        word_vector = 'word_vectors'
        sentence_vectors = pd.read_parquet(DATA_DIR / "sentence_vectors.parquet")[['description', 'word_vectors']]
        sentence_vectors = sentence_vectors[sentence_vectors["description"].notna()].drop_duplicates(subset=["description"])
        # merge descriptions and topics
        info = sentence_vectors.merge(info, left_on="description", how="inner", right_on="description")
        topics = pd.read_parquet(DATA_DIR / "descriptions_biobert_None_128.parquet")[['description', 'cluster_None', 'olr_cluster_None']]
        info = info.merge(topics, left_on="description", how="inner", right_on="description")
        info = info.sample(frac=0.1, random_state=40)
    
    helper = info.dropna(subset = [word_vector]).reset_index()

    # UMAP plot for ChEMBL embeddings
    reducer = cuml.UMAP(n_components=5, n_neighbors=15, min_dist=0.1, random_state=42)

    embeddings = np.array(helper[word_vector].to_list())
    embedding = reducer.fit_transform(embeddings)

    addition = pd.DataFrame(embedding, columns=["x", "y", "z1", "z2", "z3"])

    # UMAP_SI()
    UMAP_main_text()
    # UMAP_topics()