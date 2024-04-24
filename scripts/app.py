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
from scipy.special import expit


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

    diagnoses = re.split(r"\d+\)\s(?!\n)|\d+\.\s(?!\n)", output)
    diagnoses = [diagnosis.strip().split('\n')[0] for diagnosis in diagnoses]

    # print(f"\n{diagnoses}")

    diagnoses_filtered = diagnoses[1:6]
    return diagnoses_filtered


def compute_similarity(ground_truth, diagnosis, em_extractor):

    em1 = em_extractor.encode(ground_truth, convert_to_tensor=True)
    em2 = em_extractor.encode(diagnosis, convert_to_tensor=True)

    cosine_similarity = util.cos_sim(em1, em2)
    return cosine_similarity.item()


def get_model_benchmark(diagnoses, ground_truth, embedding_extractor):

    ground_truth = ground_truth.strip().lower()

    diagnoses = _get_diagnoses(diagnoses)

    perfect_match = False

    if len(diagnoses) > 0:

        metrics = []
        prev_diagnoses = {}
        idx = 0

        while idx < len(diagnoses) and not perfect_match:

            diagnosis = diagnoses[idx]

            if len(diagnosis) > 0:

                diagnosis = diagnosis.strip().lower()

                metric = compute_similarity(ground_truth, diagnosis, embedding_extractor)

                if metric > 0.95:  # account for non-deterministic processes
                    perfect_match = True

                if diagnosis in prev_diagnoses:
                    metric /= 2 ** prev_diagnoses[diagnosis]
                prev_diagnoses[diagnosis] = prev_diagnoses.get(diagnosis, 0) + 1

                print(f"Diagnosis: {diagnosis} Metric: {metric}")

            else:
                metric = 0

            metrics.append(metric)

            idx += 1

        metrics = np.array(metrics)

        metric_ranks = np.arange(1, len(metrics) + 1)
        metric_weights = expit(metrics) / metric_ranks

        weighted_similarity = np.average(metrics, weights=metric_weights)

    else:

        weighted_similarity = 0

    return weighted_similarity


def get_output(chain, system_prompt, case, query):

    output = chain.run(system_prompt=system_prompt,
                       case=case,
                       query=query)
    return output


if __name__ == "__main__":
    # tokenizer_biom, model_biom =
    # get_model_tokenizer("BioMistral/BioMistral-7B")

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
    query = "Provide five diagnoses. These should be sorted by likelihood. Always provide the most likely diagnosis first."

    chain = LLMChain(llm=hf_pipeline_meditron,
                     prompt=prompt)

    cases = json.load(open("data/cases.json", "r"))

    num_trials = 5
    total_metrics = {}

    for case_num, info in cases.items():

        trial_metrics = []

        for trial in range(num_trials):

            print(f"\nCase: {info['diagnosis']} Trial: {trial + 1}")
            output = get_output(chain, system_prompt, info["description"], query)
            metric = get_model_benchmark(output, info["diagnosis"], embedding_extractor)

            trial_metrics.append(metric)
            print(f"Weighted Metric: {metric}")

        average_weighted_metric = round(sum(trial_metrics) / num_trials, 4)
        print(f"\nAverage Weighted Metric: {average_weighted_metric}")
        total_metrics[case_num] = average_weighted_metric

    print(total_metrics)
