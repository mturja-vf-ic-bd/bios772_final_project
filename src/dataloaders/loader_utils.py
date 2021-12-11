import torch
from src.utils.data_utils import template_to_idx_mapping


def cut_templates_and_join(x, index_list):
    """
    Cuts specific templates from index_list and concatenate them
    :param x: torch Tensor
    :return:
    """
    temp_cuts = []
    for idx_range in index_list:
        temp_cuts.append(x[:, idx_range[0]: idx_range[1]])
    return torch.cat(temp_cuts, dim=-1)


def cut_templates(x, template_list):
    index_list = [template_to_idx_mapping[t] for t in template_list]
    return cut_templates_and_join(x, index_list)


def get_length_of_cuts(template_list):
    return sum([template_to_idx_mapping[temp][1] - template_to_idx_mapping[temp][0]
                for temp in template_list])


def get_lengths_of_template(template_name):
    idx = template_to_idx_mapping[template_name]
    return idx[1] - idx[0]


def make_dict_from_template_names(x, template_names):
    index_dict = {}
    for t in template_names:
        index_dict[t] = template_to_idx_mapping[t]
    temp_cuts = {}
    for t, idx_range in index_dict.items():
        temp_cuts[t] = x[:, idx_range[0]: idx_range[1]]
    return temp_cuts