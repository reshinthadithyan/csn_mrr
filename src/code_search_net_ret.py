from ast import parse
import logging
import argparse
from tqdm import tqdm
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModel
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset_for_eval(dataset_idt: str, subset: str, candidate_size: str = 1000):
    """
    Loads a dataset from the hf datasets library and corresponding candidates.
    :param dataset_idt: dataset id corresponding to the dataset.
    :param subset: subset of the dataset to evaluate.
    :param candidate_size: size of the candidate space.
    """
    dataset = load_dataset(dataset_idt, subset, split="test", streaming=True)
    dataset = dataset.take(candidate_size)
    return dataset


class RetDataset(Dataset):
    def __init__(self, dataset, tokenizer: AutoTokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.dataset_list = list(dataset)

    def __len__(self):
        return len(list(self.dataset))

    def __getitem__(self, index: int):
        datapoint = self.dataset_list[index]
        code, nl_desc = datapoint["whole_func_string"], datapoint["func_documentation_string"]
        return code, nl_desc


def squeeze_tree(tensor_data):
    return {k: tensor_data[k].squeeze(1) for k in tensor_data}


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(
        ~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(
        dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="microsoft/graphcodebert-base")
    parser.add_argument("--dataset", type=str, default="code_search_net")
    parser.add_argument("--subset", type=str, default="python")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--candidate_size", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    model_name = args.model
    batch_size = args.batch_size
    candidate_size = args.candidate_size

    accelerator = Accelerator()
    device = accelerator.device

    logger.info(f" Starting to load dataset ")
    test_dataset = load_dataset_for_eval(
        args.dataset, args.subset, args.candidate_size)

    logger.info("Loading tokenizer to memory..")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def preprocess_ind(datapoint):
        code, nl_desc = datapoint["whole_func_string"], datapoint["func_documentation_string"]
        tokenized_code, tokenized_nl_desc = tokenizer(code, padding="max_length", truncation=True, return_tensors='pt'), tokenizer(
            nl_desc,  padding="max_length", truncation=True, return_tensors='pt')
        return tokenized_code, tokenized_nl_desc
    test_torch_dataset = RetDataset(test_dataset, tokenizer)
    test_dataloader = DataLoader(test_torch_dataset, batch_size=batch_size)

    logger.info("Starting to run Training samples...")
    logger.info(f"Number of candidates : {len(list(test_torch_dataset))}")
    logger.info(f"Running with batch size : {batch_size}")

    logger.info(f"Loading the model into the memory.")
    model = AutoModel.from_pretrained(model_name)

    # Loading model to accelerator
    model, test_dataload = accelerator.prepare(model, test_dataloader)

    ranks = []
    code_emb_list = []
    nl_emb_list = []
    for batch in tqdm(test_dataloader, total=int(candidate_size/args.batch_size)):
        code, nl_desc = [], []
        for ind in batch:
            code.append(ind[0])
            nl_desc.append(ind[1])
        code = tokenizer(code, padding="max_length",
                         truncation=True, return_tensors='pt').to(device)
        nl_desc = tokenizer(nl_desc, padding="max_length",
                            truncation=True, return_tensors='pt').to(device)

        code_emb = model(code["input_ids"], code["attention_mask"])
        nl_emb = model(nl_desc["input_ids"], nl_desc["attention_mask"])
        code_emb = mean_pooling(code_emb[0], code['attention_mask'])
        nl_emb = mean_pooling(nl_emb[0], nl_desc['attention_mask'])
        code_vecs = code_emb.cpu().detach()
        nl_vecs = nl_emb.cpu().detach()
        code_emb_list.append(code_vecs)
        nl_emb_list.append(nl_emb)

    del model
    logger.info("Completed encoding to space...")
    nl_vecs = torch.cat(nl_emb_list, dim=0).detach().numpy()
    code_vecs = torch.cat(code_emb_list, dim=0).detach().numpy()

    scores = np.matmul(nl_vecs, code_vecs.T)
    for i in range(len(scores)):
        score = scores[i, i]
        rank = 1
        for j in range(len(scores)):
            if i != j and scores[i, j] >= score:
                rank += 1
        ranks.append(1/rank)
        print(ranks)
    print('eval_mrr: ', float(np.mean(ranks)))
