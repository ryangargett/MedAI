import os
from transformers import AutoTokenizer, AutoModel


def get_model_tokenizer(model_id, model_path):

    if os.path.exists(model_path) is False:
        print(f"Local model doesn't exist, downloading model from huggingface....")
        os.makedirs(model_path)

        print(f"Downloading and saving tokenizer....")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(model_path)

        print(f"Downloading and saving model....")
        model = AutoModel.from_pretrained(model_id)
        model.save_pretrained(model_path)

        print(f"Model and tokenizer saved successfully.")

    print(f"Loading model and tokenizer....")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    return tokenizer, model


if __name__ == "__main__":
    tokenizer, model = get_model_tokenizer("BioMistral/BioMistral-7B", "BioMistral")
