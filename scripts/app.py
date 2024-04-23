import torch
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import numpy as np
from sentence_transformers import SentenceTransformer, util
import json


def _get_bnb_config():

    config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_compute_dtype=torch.float32
    )

    return config


def get_model_tokenizer(model_id):

    model_config = _get_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 device_map="auto",
                                                 quantization_config=model_config,)

    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                              add_bos_token=True)

    if not os.path.exists(model_id):
        os.makedirs(model_id)

        model.save_pretrained(model_id, safe_serialization=False)
        tokenizer.save_pretrained(model_id)

    return tokenizer, model


def _get_diagnoses(diagnoses):

    message_components = diagnoses.split("Diagnoses:")

    if len(message_components) > 1:
        output = "Diagnoses:".join(message_components[1:])
    else:
        output = message_components[1]

    print(f"\noutput: {output}")

    diagnoses = re.split(r"^\d+\.", output, flags=re.MULTILINE)
    # diagnoses = [diagnosis.strip().split('\n')[0] for diagnosis in diagnoses]  # Account for any unnecessary generation
    diagnoses = [diagnosis.strip().split('\n')[0] for diagnosis in diagnoses]  # Remove any invalid diagnoses (e.g. ;)

    diagnoses_filtered = diagnoses[1:4]
    return diagnoses_filtered


def compute_similarity(ground_truth, diagnosis, em_extractor):

    em1 = em_extractor.encode(ground_truth, convert_to_tensor=True)
    em2 = em_extractor.encode(diagnosis, convert_to_tensor=True)

    cosine_similarity = util.cos_sim(em1, em2)
    return cosine_similarity.item()


def get_model_benchmark(diagnoses, ground_truth, embedding_extractor):

    diagnoses = _get_diagnoses(diagnoses)
    print(f"\nGround Truth: {ground_truth}")

    if len(diagnoses) > 0:

        metrics = []

        for diagnosis in diagnoses:
            if len(diagnosis) > 0:
                metric = compute_similarity(ground_truth, diagnosis, embedding_extractor)
            else:
                metric = 0
            print(f"Diagnosis: {diagnosis}; Metric: {metric}")
            metrics.append(metric)

        weights = 1 / np.arange(1, len(metrics) + 1)

        weighted_similarity = np.average(metrics, weights=weights)

    else:

        weighted_similarity = 0

    print(f"Weighted Similarity: {weighted_similarity}")
    return weighted_similarity


def get_output(chain, system_prompt, case, query):

    output = chain.run(system_prompt=system_prompt,
                       case=case,
                       query=query)
    return output


if __name__ == "__main__":
    # tokenizer_biom, model_biom = get_model_tokenizer("BioMistral/BioMistral-7B")
    tokenizer_meditron, model_meditron = get_model_tokenizer("epfl-llm/meditron-7b")

    embedding_extractor = SentenceTransformer("NeuML/pubmedbert-base-embeddings",
                                              device="cuda")

    pipe_meditron = pipeline("text-generation",
                             model=model_meditron,
                             tokenizer=tokenizer_meditron,
                             max_new_tokens=100,
                             temperature=0.3,
                             do_sample=True)

    hf_pipeline_meditron = HuggingFacePipeline(pipeline=pipe_meditron)

    template = "{system_prompt}\n\nCase: {case}\n\nQuery: {query}\n\nDiagnoses:"

    prompt = PromptTemplate(input_variables=["system_prompt", "case", "query"],
                            template=template)

    system_prompt = "You are a helpful medical assistant. You will be provided and asked about a complicated clinical case; read it carefully and then provide a concise DDx."
    query = "Provide three diagnoses. These should be sorted by likelihood. Always provide the most likely diagnosis first."

    chain = LLMChain(llm=hf_pipeline_meditron,
                     prompt=prompt)

    cases = json.load(open("data/cases.json", "r"))

    for case_num, info in cases.items():
        output = get_output(chain, system_prompt, info["description"], query)
        metric = get_model_benchmark(output, info["diagnosis"], embedding_extractor)
