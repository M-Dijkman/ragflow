import chembl_downloader
import pandas as pd
from bertopic import BERTopic


def get_assays(df):
    df_aids = [entry.split(";") for entry in df.AID.unique().tolist()]
    flat_aids = [item for sublist in df_aids for item in sublist]
    unique_aids = set(flat_aids)
    unique_chembl_aids = [item for item in unique_aids if item.startswith("CHEMBL")]
    
    return unique_chembl_aids


def add_topics(chembl_aids):
    df = pd.read_parquet("assayctx/bert/descriptions_biobert_None_128.parquet", columns=["chembl_id", "olr_cluster_None"])
    df = df.loc[df["chembl_id"].isin(chembl_aids)]
    
    return df


def topic_information(chembl_aids):
    df = pd.read_parquet("AssayCTX/bert/descriptions_biobert_None_128.parquet", columns=["chembl_id", "olr_cluster_None"])
    df = df.loc[df["chembl_id"].isin(chembl_aids)]
    topic_model = BERTopic.load("AssayCTX/bert/saved_topicmodel_biobert_None_None_128_dir")
    info = topic_model.get_topic_info()[["Topic", "Representation"]]
    topic_info = df.merge(info, left_on="olr_cluster_None", right_on="Topic", how="left")
    
    return topic_info


def query(chembl_aids):
    path = chembl_downloader.download_extract_sqlite(version="34")
    sql = f"""
    SELECT
        ASSAYS.chembl_id,
        ASSAYS.assay_type
    FROM ASSAYS
    WHERE ASSAYS.chembl_id IN {tuple(chembl_aids)}
    """
    df = chembl_downloader.query(sql)
    df = df.drop_duplicates()
    
    return df


if __name__ == "__main__":
    # Laad de originele dataset
    df = pd.read_csv("AssayCTX/data/hERG_Dataset.tsv", sep="\t")

    # Ophalen van CHEMBL ID's
    chembl_aids = get_assays(df)

    # Query voor de chembl_info
    chembl_info = query(chembl_aids)

    # Wegschrijven van chembl_info naar CSV
    chembl_info.to_csv("chembl_info.csv", index=False)

    # Merge de chembl_info met de originele dataset op chembl_id
    merged_df = df.merge(chembl_info, left_on="AID", right_on="chembl_id", how="left")

    # Filter rijen waar assay_type gelijk is aan 'F'
    filtered_df = merged_df[merged_df["assay_type"] != "F"]

    # Schrijf het gefilterde bestand weg
    filtered_df.to_csv("AssayCTX/data/hERG_Dataset_filtered.tsv", sep="\t", index=False)
#deze methode werkt best goed. Enige wat 'mis gaat' is dat in het nieuwe bestand sommige rijen zijn blijven staan, omdat deze in de originele dataset meerdere chemblID's hebben en de code hier niet weet wat er mee gedaan moet worden