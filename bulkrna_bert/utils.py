import torch
import math


def mask_input_ids(
    input_ids: torch.Tensor,
    mask_percentage=0.15,
    mask_token_id=0,
    max_bin_id=63,
    only_mask=False,
) -> dict:
    if mask_percentage == 0:
        return {
            "input_ids": input_ids,
            "labels": torch.full_like(input_ids, fill_value=-100),
        }
    input_ids = input_ids.clone()
    n_tokens = input_ids.size(0)
    num_to_mask = math.ceil(mask_percentage * n_tokens)

    # Randomly select positions
    selected_indices = torch.randperm(n_tokens)[:num_to_mask]

    # 80% mask, 10% random, 10% unchanged
    n_mask = int(0.8 * num_to_mask)
    n_random = int(0.1 * num_to_mask)
    n_unchanged = num_to_mask - n_mask - n_random

    perm = torch.randperm(num_to_mask)
    mask_indices = selected_indices[perm[:n_mask]]
    random_indices = selected_indices[perm[n_mask : n_mask + n_random]]
    # unchanged_indices = selected_indices[perm[n_mask + n_random :]]

    # Prepare labels
    labels = torch.full_like(input_ids, -100)
    labels[selected_indices] = input_ids[selected_indices]

    if only_mask:
        input_ids[selected_indices] = mask_token_id
    else:
        # Apply masking
        input_ids[mask_indices] = mask_token_id
        input_ids[random_indices] = torch.randint(
            0, max_bin_id + 1, (n_random,), dtype=input_ids.dtype
        )

    return {"input_ids": input_ids, "labels": labels}


def split_train_val_test_indices(dataset_labels, num_to_select=5):
    all_train_indices = []
    all_val_indices = []
    all_test_indices = []

    for i in torch.unique(dataset_labels).tolist():
        indices = torch.where(dataset_labels == i)[0]
        select_indices = indices[torch.randperm(indices.shape[0])[: num_to_select * 2]]
        val_indices = select_indices[:num_to_select]
        test_indices = select_indices[num_to_select:]
        train_indices = indices[
            ~torch.isin(indices, torch.concat([val_indices, test_indices]))
        ]
        all_train_indices.append(train_indices)
        all_val_indices.append(val_indices)
        all_test_indices.append(test_indices)

    all_train_indices = torch.concat(all_train_indices)
    all_val_indices = torch.concat(all_val_indices)
    all_test_indices = torch.concat(all_test_indices)
    return all_train_indices, all_val_indices, all_test_indices


if __name__ == "__main__":
    from bulkrna_bert.tokenizer import BinnedOmicTokenizer
    import pandas as pd
    import numpy as np

    repo = "InstaDeepAI/BulkRNABert"
    df = pd.read_csv(
        "/mnt/hdd/Shervin/Thesis/bulkrna_bert/data/tcga_all.csv", index_col=0
    )
    expressions = df.values[:, :-1].astype(np.float32)
    expressions = np.log10(expressions + 1)

    tokenizer = BinnedOmicTokenizer.from_pretrained(repo)
    inputs = tokenizer.batch_encode_plus(expressions, return_tensors="pt")

    a = mask_input_ids(inputs["input_ids"][0])
    a

    labels = a["labels"]
    input_ids = a["input_ids"]

    indices = torch.where(labels != -100)[0]
    print(f"input_ids:\t {input_ids[indices[:10]]}")
    print(f"labels:\t {labels[indices[:10]]}")
