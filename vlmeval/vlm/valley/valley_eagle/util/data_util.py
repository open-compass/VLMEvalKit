import copy
import re
from typing import Dict, Sequence

import torch
import transformers
from PIL import Image
from transformers import CLIPImageProcessor, StoppingCriteria
from .. import conversation as conversation_lib
import ast
import math

# from valley.constants import *
# from valley.util.config import *
from .config import (
    DEFAULT_GANDALF_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VI_END_TOKEN,
    DEFAULT_VI_START_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    GANDALF_TOKEN_INDEX,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    SEQ_MAX_LEN
)
SPLIT_TOKEN = "<SPLIT_TOKEN>"


def collate_wrapper(batch):
    try:
        image_list = [b[0] for b in batch]
        prompt_list = [b[2] for b in batch]
        # input_ids = pad_sequence(prompt_list, padding_value = 0, batch_first = True)
        conv_list = [b[3] for b in batch]
        save_id_list = [b[4] for b in batch]
        label_list = [b[1] for b in batch]
    except Exception as e:
        prompt_list, image_list, conv_list, label_list, save_id_list = None, None, None, None, None
        print(f"error in collate_wrapper: {e} ||| all set to None")
    return prompt_list, image_list, conv_list, label_list, save_id_list


def collate_process_image_text(batch, tokenizer, image_processor):
    batch_input_ids, batch_image, conv_list, label_list, save_id_list = batch
    input_ids = torch.stack(batch_input_ids, dim=0)
    videos = []
    for this_batch_images in batch_image:
        if (
            ".mp4" not in save_id_list[0] and ".avi" not in save_id_list[0]
        ):  # if not a video file, do image list process func
            video = image_processor.preprocess(this_batch_images, return_tensors="pt")["pixel_values"]
            videos.append(video)
        else:
            videos.append(this_batch_images)
    return input_ids, videos, conv_list, label_list, save_id_list


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


# for finetune


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.args.should_save:
        if getattr(trainer.args, "lora", None):
            trainer.model.save_pretrained(output_dir)
            if trainer.args.tune_mm_mlp_adapter:
                trainer.model.base_model.model.save_pretrained(output_dir)
        else:
            state_dict = trainer.model.state_dict()
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict
            trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def tokenizer_image_token(
    prompt,
    tokenizer,
    image_token_index=IMAGE_TOKEN_INDEX,
    gandalf_token_index=GANDALF_TOKEN_INDEX,
    return_tensors=None,
):
    def split_with_token(string, token):
        result = string.split(token)
        for i in range(len(result) - 1):
            result.insert(i * 2 + 1, token)
        return result

    if len(prompt) > SEQ_MAX_LEN:
        # This error will be caught by the __getitem__ method in LazySupervisedDataset within valley/data/dataset.py,
        # and it will then randomly select another valid data item to return.
        raise ValueError("sequence is too long !!!")

    prompt_chunks = split_with_token(prompt, DEFAULT_IMAGE_TOKEN)
    prompt_chunks = sum([split_with_token(chunk, DEFAULT_GANDALF_TOKEN) for chunk in prompt_chunks], [])
    input_ids, offset = ([tokenizer.bos_token_id], 1) if getattr(tokenizer,'bos_token',None) else ([], 0)
    token2index = {DEFAULT_IMAGE_TOKEN: image_token_index, DEFAULT_GANDALF_TOKEN: gandalf_token_index}
    for chunk in prompt_chunks:
        if chunk in token2index:
            input_ids.append(token2index[chunk])
        else:
            chunk_ids = tokenizer(chunk).input_ids
            # For Qwen2-7B, bos token exists but does not appear in the beginning
            if chunk_ids[0] != getattr(tokenizer,'bos_token_id', None):
                offset = 0
            input_ids.extend(chunk_ids[offset:])

    # prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]
    # def insert_separator(X, sep):
    #     return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    # input_ids = []
    # offset = 0
    # if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
    #     offset = 1
    #     input_ids.append(prompt_chunks[0][0])

    # for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
    #     input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers, only_mask_system=False):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    if not only_mask_system:
        for tokenized_len, speaker in zip(tokenized_lens, speakers):
            if speaker == "human":
                target[cur_idx + 2: cur_idx + tokenized_len] = IGNORE_INDEX
            cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"].strip()
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    conversations: Sequence[dict],
    img_num,
    data_args,
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return conversations

    for sentence in conversations:
        if data_args.model_class in ["valley-product", "valley-gandalf", "tinyvalley", "valley-product-mistral"]:
            if DEFAULT_VIDEO_TOKEN in sentence["value"]:
                if data_args.use_special_start_end_token:
                    video_replace_token = (
                        DEFAULT_VI_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_VI_END_TOKEN
                    ) * img_num
                else:
                    video_replace_token = DEFAULT_IMAGE_TOKEN * img_num
                # video_replace_token = ' '.join(f'Frame {i}: {DEFAULT_IMAGE_TOKEN}' for i in range(img_num))
                sentence["value"] = sentence['value'].replace(DEFAULT_VIDEO_TOKEN, '').strip()
                sentence["value"] = video_replace_token + '\n' + sentence["value"]
            else:
                segs = re.split(DEFAULT_IMAGE_TOKEN, sentence["value"])
                if data_args.use_special_start_end_token:
                    sentence["value"] = (
                        DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                    ).join(segs[: img_num + 1]) + "".join(segs[img_num + 1:])
                else:
                    sentence["value"] = DEFAULT_IMAGE_TOKEN.join(segs[: img_num + 1]) + "".join(
                        segs[img_num + 1:]
                    )
        elif data_args.model_class in ["valley-video", "valley-video-mistral"]:
            if DEFAULT_IMAGE_TOKEN in sentence["value"] or DEFAULT_VIDEO_TOKEN in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = sentence["value"].replace(DEFAULT_VIDEO_TOKEN, "").strip()
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(
                        DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>"
                    )
        else:
            raise Exception("unknown model class")

    return conversations


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    inference: bool = False,
    only_mask_system: bool = False,
) -> Dict:
    '''
    FIXME: support only_mask_system=True; check tokenizer; unwrap sources
    '''
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    sources = [sources]

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        if inference:
            conv.append_message(conv.roles[1], None)
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids.squeeze(0),
        labels=targets.squeeze(0),
    )


def preprocess_mistral(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    inference: bool = False,
    only_mask_system: bool = False,
) -> Dict:
    """
    FIXME: support only_mask_system=True; check tokenizer; unwrap sources
    """
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    sources = [sources]

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        if inference:
            source.pop(-1)
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        if inference:
            conv.append_message(conv.roles[1], None)
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    # assert (input_ids == 1).sum() == 2 and input_ids.shape[0] ==1
    # input_ids = input_ids[:,1:]
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.MISTRAL

    # Mask targets
    sep = "[/INST]"
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        if not only_mask_system:
            target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length and not only_mask_system and not inference:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v0(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    inference: bool = False,
    only_mask_system: bool = False,
) -> Dict:
    """
    FIXME: check tokenizer; unwrap sources
    """
    sources = [sources]
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers, only_mask_system=only_mask_system)

    return dict(input_ids=input_ids, labels=targets)


def preprocess_v1(
    source,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    inference: bool = False,
    only_mask_system: bool = False,
) -> Dict:
    """
    FIXME: support only_mask_system=True
    """
    conv = conversation_lib.default_conversation.copy()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    if roles[source[0]["from"]] != conv.roles[0]:
        # Skip the first one if it is not from human
        source = source[1:]

    conv.messages = []
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        assert role == conv.roles[j % 2], f"{j}"
        conv.append_message(role, sentence["value"])
    if inference:
        conv.append_message(conv.roles[1], None)

    conversation = conv.get_prompt()

    # Mask targets
    rounds = conversation.split(conv.sep2)

    input_ids_ = torch.tensor([1], dtype=torch.int64)
    targets_ = torch.tensor([-100], dtype=torch.int64)
    for i, rou in enumerate(rounds):
        if rou == "":
            continue
        if (not inference) or (i < (len(rounds) - 1)):
            rou += conv.sep2
        if has_image:
            cur_input_ids_ = tokenizer_image_token(rou, tokenizer, return_tensors="pt")[1:]
            input_ids_ = torch.cat([input_ids_, cur_input_ids_], dim=0)
            if only_mask_system:
                mask_len = len(
                    tokenizer_image_token(re.sub(rf"{conv.roles[0]}:[\s\S]*", f"{conv.roles[0]}:", rou), tokenizer)[1:]
                )
            else:
                mask_len = len(
                    tokenizer_image_token(re.sub(rf"{conv.roles[1]}:[\s\S]*", f"{conv.roles[1]}:", rou), tokenizer)[1:]
                )
            # targets_ = torch.cat([targets_, torch.tensor([-100] * mask_len), cur_input_ids_[mask_len:]], dim=0)
            targets_ = torch.cat([targets_, torch.tensor([-100] * mask_len), cur_input_ids_[mask_len:]], dim=0)
        else:
            cur_input_ids_ = tokenizer(rou, return_tensors="pt")["input_ids"][0, 1:]
            input_ids_ = torch.cat([input_ids_, cur_input_ids_], dim=0)
            mask_len = len(tokenizer(re.sub(rf"{conv.roles[1]}:[\s\S]*", f"{conv.roles[1]}:", rou))["input_ids"][1:])
            # targets_ = torch.cat([targets_, torch.tensor([-100] * mask_len), cur_input_ids_[mask_len:]], dim=0)
            targets_ = torch.cat([targets_, torch.tensor([-100] * mask_len), cur_input_ids_[mask_len:]], dim=0)
    return {"input_ids": input_ids_, "labels": targets_}


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess_uninstruct_text_image(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    content = sources["content"]

    input_ids_ = torch.tensor([1], dtype=torch.int64) if tokenizer.bos_token else torch.tensor([], dtype=torch.int64)
    targets_ = torch.tensor([-100], dtype=torch.int64) if tokenizer.bos_token else torch.tensor([], dtype=torch.int64)
    cur_input_ids_ = tokenizer_image_token(content, tokenizer, return_tensors="pt")[1:]
    input_ids_ = torch.cat([input_ids_, cur_input_ids_], dim=0)
    targets_ = torch.cat([targets_, cur_input_ids_[:]], dim=0)

    return {"input_ids": input_ids_, "labels": targets_}


def preprocess_text(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:

    content = sources["content"]
    if len(content) > SEQ_MAX_LEN:
        # This error will be caught by the __getitem__ method in LazySupervisedDataset within valley/data/dataset.py,
        # and it will then randomly select another valid data item to return.
        raise ValueError("sequence is too long !!!")

    input_tokens = []
    bos_token = [tokenizer.bos_token] if tokenizer.bos_token else []  # suppor qwen2
    for sub_text in content.split(SPLIT_TOKEN):
        input_tokens.extend(bos_token + tokenizer.tokenize(sub_text) + [tokenizer.eos_token])
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(input_tokens))
    targets = input_ids.clone()

    return {"input_ids": input_ids, "labels": targets}


def preprocess_qwen2(
        source,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
        inference: bool = False,
        only_mask_system: bool = False,
):
    '''
      "chat_template":
      "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}
      {{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' +
      message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
      {% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
    '''
    conv = conversation_lib.default_conversation.copy()
    assert conv.sep_style == conversation_lib.SeparatorStyle.QWEN2
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    # Apply prompt templatess
    if roles[source[0]["from"]] != conv.roles[0]:
        # Skip the first one if it is not from human
        source = source[1:]
    messages = []
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        assert role == conv.roles[j % 2], f"{j}"
        messages.append({"role":role, "content":sentence["value"]})
    conversation = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=inference)
    # Mask targets
    rounds = conversation.split(conv.sep2)
    input_ids_ = torch.tensor([], dtype=torch.int64)
    targets_ = torch.tensor([], dtype=torch.int64)
    for i, rou in enumerate(rounds):
        if rou == "":
            continue
        if (not inference) or (i < (len(rounds) - 1)):
            rou += conv.sep2
        if has_image:
            cur_input_ids_ = tokenizer_image_token(rou, tokenizer, return_tensors='pt')
            input_ids_ = torch.cat([input_ids_, cur_input_ids_], dim=0)
            if only_mask_system:
                mask_len = len(tokenizer_image_token(re.sub(rf'{conv.roles[0]}\n[\s\S]*', f'{conv.roles[0]}:', rou),
                                                     tokenizer))
            else:
                mask_len = len(tokenizer_image_token(re.sub(rf'{conv.roles[1]}\n[\s\S]*', f'{conv.roles[1]}:', rou),
                                                     tokenizer))
            targets_ = torch.cat([targets_, torch.tensor([-100] * mask_len), cur_input_ids_[mask_len:]], dim=0)
        else:
            cur_input_ids_ = tokenizer(rou, return_tensors='pt')["input_ids"][0, :]
            input_ids_ = torch.cat([input_ids_, cur_input_ids_], dim=0)
            mask_len = len(tokenizer(re.sub(rf'{conv.roles[1]}\n[\s\S]*', rf'{conv.roles[1]}:', rou))["input_ids"][:])
            # targets_ = torch.cat([targets_, torch.tensor([-100] * mask_len), cur_input_ids_[mask_len:]], dim=0)
            targets_ = torch.cat([targets_, torch.tensor([-100] * mask_len), cur_input_ids_[mask_len:]], dim=0)
    return {"input_ids": input_ids_, "labels": targets_}


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    only_mask_system: bool = False,
    inference: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    assert conversation_lib.default_conversation.version in [
        "v0", "v1", "mistral", "llama_2", "plain", 'qwen2','gemma2'
    ]
    # v0 is for vicuna-v0, sep is '###'
    # v1 is for vicuna-v1.x, sep is ' ', sep2 is '</s>'
    # mistral is for mistral, sep is [INST]
    # llama_2 is for llama2, sep is [INST]
    # plain is for pretraining, no chat tamplete
    # please refer to file examples/valleyproduct/valley/conversation.py for details
    if isinstance(sources, dict):
        if sources["preprocess_mode"] == "uninstruct_text_image":
            return preprocess_uninstruct_text_image(sources, tokenizer)
        elif sources["preprocess_mode"] == "puretext":
            return preprocess_text(sources, tokenizer)

    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(
            sources, tokenizer, has_image=has_image, inference=inference, only_mask_system=only_mask_system
        )
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.QWEN2:
        return preprocess_qwen2(
            sources, tokenizer, has_image=has_image, inference=inference, only_mask_system=only_mask_system
        )
    if conversation_lib.default_conversation.version == "v0":
        return preprocess_v0(
            sources, tokenizer, has_image=has_image, inference=inference, only_mask_system=only_mask_system
        )
    if conversation_lib.default_conversation.version == "v1":
        return preprocess_v1(
            sources, tokenizer, has_image=has_image, inference=inference, only_mask_system=only_mask_system
        )
    if conversation_lib.default_conversation.version == "mistral":
        return preprocess_mistral(
            sources, tokenizer, has_image=has_image, inference=inference, only_mask_system=only_mask_system
        )
    if conversation_lib.default_conversation.version.startswith("v1"):
        print(
            f"you'd better change your conversation version, current version is "
            f"{conversation_lib.default_conversation.version}"
        )
        return preprocess_v1(
            sources, tokenizer, has_image=has_image, inference=inference, only_mask_system=only_mask_system
        )


def find_closest_aspect_ratio(aspect_ratio, min_tile_num, max_tile_num, width, height, tiled_image_size):
    """
    Find the closest aspect ratio from a min tiles' number and a max tiles' number to the current image's aspect ratio.
    An example usage:
    find_closest_aspect_ratio(1.5, 1, 6, 1200, 800, 1024)

    This will return the aspect ratio that is closest to 1.5, considering the image dimensions  and preferring a larger
    relative area to the 'image_size'.
    In case of a tie, the ratio that results in a larger relative area compared to the original image size is chosen.

    Args:
    aspect_ratio (float): The current image's aspect ratio (width divided by height), e.g., 1.5.
    max_tile_num (int): crop min tiles of the image
    max_num (int): crop min tiles of the image
    width (int): The width of the current image, e.g., 1200.
    height (int): The height of the current image, e.g., 800.
    tiled_image_size (int): the tile size , e.g, 336.

    Returns:
        Tuple[int, int]: The aspect ratio closest to the current image's aspect ratio
        based on the criteria, e.g., (16, 9).
    """
    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_tile_num, max_tile_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_tile_num and i * j >= min_tile_num)

    # sort by aera
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the best ratio
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            # choose the larger area, if the aspect ratio is the same like 2:3 and 4:6.
            # And in this case(2:3 and 4:6), if the image aera is larger than 1/2 sum of all tiles aera,
            # then choose 4:6,
            # because the target_ratios is sorted, 4:6 is behind 2:3, the final ratio will be 4:6.
            all_tile_aera_sum = tiled_image_size * tiled_image_size * ratio[0] * ratio[1]
            if area > 0.5 * all_tile_aera_sum:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, tiled_image_size=448, use_thumbnail=False):
    """
    Processes an image dynamically based on its aspect ratio and specified parameters,
    splitting it into sub-images or creating a thumbnail as needed.

    Example:
    >>> from PIL import Image
    >>> img = Image.open('example.jpg')
    >>> processed_imgs = dynamic_preprocess(img, min_num=1, max_num=6, image_size=448, use_thumbnail=True)

    Args:
        image (PIL.Image.Image): Input image to be processed.
        min_num (int): Minimum product of width and height for aspect ratio consideration.
        max_num (int): Maximum product of width and height for aspect ratio consideration.
        image_size (int): Target size for resizing images.
        use_thumbnail (bool): Whether to append a thumbnail of the original image if multiple sub-images are generated.

    Returns:
        List[PIL.Image.Image]: A list of processed images after resizing and/or splitting, with an optional thumbnail.

    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, min_num, max_num, orig_width, orig_height, tiled_image_size)

    # calculate the target width and height
    target_width = tiled_image_size * target_aspect_ratio[0]
    target_height = tiled_image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))

    # save_resize_ratio
    width_resize_ratio = target_width / orig_width
    height_resize_ratio = target_height / orig_height

    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // tiled_image_size)) * tiled_image_size,
            (i // (target_width // tiled_image_size)) * tiled_image_size,
            ((i % (target_width // tiled_image_size)) + 1) * tiled_image_size,
            ((i // (target_width // tiled_image_size)) + 1) * tiled_image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((tiled_image_size, tiled_image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio, width_resize_ratio, height_resize_ratio
