import torch
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain


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

    message_components = diagnoses.split("Diagnosis:")

    if len(message_components) > 1:
        output = "Diagnosis:".join(message_components[1:])
    else:
        output = message_components[1]

    diagnoses = re.split(r"\d+\.", output)
    diagnoses = [diagnosis.strip() for diagnosis in diagnoses]

    return diagnoses


def get_model_benchmark(diagnoses):

    diagnoses = _get_diagnoses(diagnoses)


if __name__ == "__main__":
    # tokenizer_biom, model_biom = get_model_tokenizer("BioMistral/BioMistral-7B")
    tokenizer_meditron, model_meditron = get_model_tokenizer("BioMistral/BioMistral-7B")

    # pipe_biom = pipeline("text-generation",
    #                     model=model_biom,
    #                     tokenizer=tokenizer_biom,
    #                    max_new_tokens=200)

    pipe_meditron = pipeline("text-generation",
                             model=model_meditron,
                             tokenizer=tokenizer_meditron,
                             max_new_tokens=50,
                             temperature=0.1,
                             do_sample=True)

    # hf_pipeline_biom = HuggingFacePipeline(pipeline=pipe_biom)
    hf_pipeline_meditron = HuggingFacePipeline(pipeline=pipe_meditron)

    template = "{system_prompt}\n\nCase: {case}\n\nQuery: {query}\n\nDiagnoses:"

    prompt = PromptTemplate(input_variables=["system_prompt", "case", "query"],
                            template=template)

    system_prompt = "You are a helpful medical assistant. You will be provided and asked about a complicated clinical case; read it carefully and then provide a concise DDx."
    case = "An asymptomatic 47-year-old woman with a history of recurrent melanoma presented to the hospital after routine quarterly surveillance imaging, performed after resection of right axillary melanoma, radiation therapy, and pembrolizumab therapy, had revealed new hilar and mediastinal lymphadenopathy and new pulmonary nodules. Diagnostic tests were performed."
    query = "Provide five concise diagnoses. These should be sorted by likelihood."

    chain = LLMChain(llm=hf_pipeline_meditron,
                     prompt=prompt)

    diagnosis = chain.run(system_prompt=system_prompt,
                          case=case,
                          query=query)

    print(diagnosis)

    get_model_benchmark(diagnosis)
