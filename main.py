import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from umap import UMAP


# Preprocess each .json files:
def time_difference(start, end):
    time_format = "%H:%M:%S.%f"
    start_dt = datetime.strptime(start, time_format)
    end_dt = datetime.strptime(end, time_format)
    diff = (end_dt - start_dt).total_seconds()
    return diff


encoder = SentenceTransformer("all-MiniLM-L6-v2")  # 384‑dim

umap_model = UMAP(
    n_neighbors=10,
    n_components=15,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
)

hdbscan_model = HDBSCAN(
    min_cluster_size=3,
    min_samples=1,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)

topic_model = BERTopic(
    embedding_model=encoder,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    calculate_probabilities=False,
    verbose=True,
)

client = QdrantClient(url="http://eng-ai-agents-qdrant-1:6333")

video_folder = Path("./videos/")
processing_folder = str(Path("./cleaned/"))
videofile = ""
for raw_path in video_folder.rglob("*"):
    if (
        raw_path.is_file()
        and raw_path.suffix.lower() == ".json"
        and not str(raw_path).find(".info") != -1
    ):
        filename = str(raw_path.name)
        print("processing file ", filename)
        file_path = str(raw_path.relative_to(video_folder))
        videofile = str(raw_path.parent) + "/" + filename.rsplit(".", 1)[0] + ".mp4"
        with open(str(video_folder) + "/" + file_path, "r") as f:
            data = json.load(f)

        original_count = len(data["captions"])

        # Filter out captions with duration < 1 second
        filtered_captions = [
            caption
            for caption in data["captions"]
            if time_difference(caption["start"], caption["end"]) >= 1.0
        ]

        # Remove every other caption to decrease redundancy
        cleaned_captions = [
            caption for idx, caption in enumerate(filtered_captions) if idx % 2 == 0
        ]

        # Fix the start and end timestamps for each entry
        for idx in range(len(cleaned_captions) - 1):
            next_start = cleaned_captions[idx + 1]["start"]
            cleaned_captions[idx]["end"] = next_start

        # Update captions in data
        data["captions"] = cleaned_captions
        print(
            f"Processed {filename}: {original_count} -> {len(cleaned_captions)} captions"
        )

        # Save the processed JSON into the new folder
        cleaned_file = processing_folder + "cleaned_" + filename
        with open(cleaned_file, "w") as f:
            json.dump(data, f, indent=2)
        INPUT_JSON = Path(cleaned_file)
        # ──────────────────────────── 1  Load captions ────────────────────────────── #

        with INPUT_JSON.open("r", encoding="utf‑8") as f:
            data = json.load(f)

        # The file can be a dict (one video) or a list (many videos);
        # in either case we want the list of caption strings.
        if isinstance(data, dict) and "captions" in data:
            captions = [c["text"] for c in data["captions"]]
        elif isinstance(data, list):
            captions = [c["text"] for item in data for c in item.get("captions", [])]
        else:
            raise ValueError("Unsupported JSON structure.")

        print(f"Loaded {len(captions)} caption lines")

        # ─────────────────────────── 2  Build a topic model ───────────────────────── #

        topics, probs = topic_model.fit_transform(captions)

        # ─────────────────────────── 3  Inspect all topics ────────────────────────── #
        topic_info: pd.DataFrame = topic_model.get_topic_info()
        topic_info = topic_info[topic_info.Topic != -1]  # drop outliers

        print("\n══════════  Discovered Topics  ══════════\n")
        for _, row in topic_info.iterrows():
            topic_id = int(row.Topic)
            size = int(row.Count)

            # Build a readable label from the first three key terms
            top_terms = row.Name.split()[:3]  # e.g. ["convolution", "neural", "image"]
            topic_label = " / ".join(top_terms).title()

            caption_indices = [i for i, t in enumerate(topics) if t == topic_id]

            print(f"[{topic_label}]  ({size} captions)")
            print(
                f"  ► Caption indices: {caption_indices[:10]}{' …' if len(caption_indices) > 10 else ''}"
            )
            print()

        # ─────────────────────────── 4  (Optional) save topics CSV ────────────────── #
        embeddings = topic_model.embedding_model.embed(
            captions
        )  # ndarray shape = (len(docs), 384)
        embed_topics = (topic_info.Name).to_numpy()
        embed_topics = [
            embed_topics[i].split("_", 1)[1] for i in range(len(embed_topics))
        ]

        dim = embeddings.shape[1]  # 384 for MiniLM
        if not client.collection_exists("bertopic_sentences"):
            client.create_collection(  # drops if exists, then creates
                collection_name="bertopic_sentences",
                vectors_config=models.VectorParams(
                    size=dim, distance=models.Distance.COSINE
                ),
            )
        # Pull out the timestamps
        points = [
            models.PointStruct(
                id=i,
                vector=embeddings[i].tolist(),  # ensure plain Python list/np array
                payload={
                    "id": i,
                    "videofile": videofile,
                    "topic_id": int(topics[i]),
                    "topic_text": embed_topics[topics[i]],
                    "text": captions[i],
                    "start": data["captions"][i]["start"],
                    "end": data["captions"][i]["end"],
                    "prob": float(probs[i]) if probs is not None else None,
                },
            )
            for i in range(0, len(captions))
        ]
        client.upsert(
            collection_name="bertopic_sentences", points=points
        )  # inserts or overwrites

# 1. Load your SentenceTransformer
import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# from sklearn.metrics.pairwise import cosine_similarity

encoder = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim

import os
import subprocess

import gradio as gr


def split_segment(input_path, start, end, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-to",
        str(end),
        "-i",
        input_path,
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return output_path


def get_answer(prompt: str) -> str:
    clip_paths = []
    full_string = ""
    question = prompt
    q_vec = encoder.encode(question, convert_to_numpy=True)
    q_vec = (q_vec / np.linalg.norm(q_vec)).tolist()  # 1‑D list

    query_response = client.query_points(  # or .search(...)
        collection_name="bertopic_sentences",
        query=q_vec,
        limit=5,  # top‑k
    )
    hits = query_response.points

    i = 0
    for h in hits:
        outp = f"clips/seg{i}.mp4"
        clip_paths.append(
            split_segment(
                h.payload["videofile"], h.payload["start"], h.payload["end"], outp
            )
        )
        i += 1

    for h in hits:
        full_string += f"{h.payload['start']}: {h.payload['text']}\n"

        # print(
        #     f"Video: {h.payload['videofile']} \n Start Time: {h.payload['start']} \n End Time: {h.payload['end']} \n Text: {h.payload['text']}"
        # )
        # print(f"[{h.score: .3f}] {h.payload['id']}")
    return clip_paths, full_string


# 3) build the UI
with gr.Blocks() as demo:
    gr.Markdown("## Chat with Your Video Library")
    with gr.Row():
        with gr.Column(scale=2):
            question = gr.Textbox(label="Your Question", placeholder="Type something…")
            answer = gr.Textbox(label="Answer", interactive=False)
            gallery = gr.Gallery(
                label="Clipped Segments", columns=2, file_types=["video"]
            )
            btn = gr.Button("Get Answer")
    btn.click(fn=get_answer, inputs=question, outputs=[gallery, answer])

# 4) launch
if __name__ == "__main__":
    demo.launch()
