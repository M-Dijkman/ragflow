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
        ASSAYS.description
    FROM ASSAYS
    WHERE ASSAYS.chembl_id IN {tuple(chembl_aids)}
    """
# hier nog een filter in sql code om assay type te filteren
    df = chembl_downloader.query(sql)
    df = df.drop_duplicates()

    return df

if __name__ == "__main__":
    #juiste input file (wat je al uit papyrus  had gehaald)
    df = pd.read_csv("AssayCTX/data/mGluR5_v1.tsv", sep="\t")
    chembl_aids = get_assays(df)
    chembl_info = query(chembl_aids)
    #assay_info = topic_information(chembl_aids)

    #chembl_info.to_csv("chembl_info.csv", index=False)
    #assay_info.to_csv("assay_info.csv", index=False)


    #chembl info joinen met dataset en wegschrijven wat je overhoudt