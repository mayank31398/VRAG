import json
import logging
import os
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
from transformers import (AutoConfig, AutoTokenizer, DPRQuestionEncoder,
                          GPT2LMHeadModel)

from .data import pad_ids

logger = logging.getLogger(__name__)
EPSILON = 1e-10


def GetUnionKL(prior_model_outputs, posterior_model_outputs):
    prior_topk_documents_ids = prior_model_outputs["topk_documents_ids"]
    posterior_topk_documents_ids = posterior_model_outputs["topk_documents_ids"]
    prior_question_embeddings = prior_model_outputs["question_embeddings"]
    posterior_question_embeddings = posterior_model_outputs["question_embeddings"]
    prior_topk_documents_embeddings = prior_model_outputs["topk_documents_embeddings"]
    posterior_topk_documents_embeddings = posterior_model_outputs["topk_documents_embeddings"]

    batch_size = len(prior_topk_documents_ids)
    topk = len(prior_topk_documents_ids[0])

    KL = 0
    for i in range(batch_size):
        all_docs_embeds = []
        s = set()
        for j in range(topk):
            s.add(prior_topk_documents_ids[i][j])
            s.add(posterior_topk_documents_ids[i][j])
        for j in s:
            if (j in prior_topk_documents_ids[i]):
                k = prior_topk_documents_ids[i].index(j)
                all_docs_embeds.append(prior_topk_documents_embeddings[i][k])
            else:
                k = posterior_topk_documents_ids[i].index(j)
                all_docs_embeds.append(
                    posterior_topk_documents_embeddings[i][k])

        all_docs_embeds = torch.tensor(all_docs_embeds).T.cuda()

        prior_logits_full = prior_question_embeddings[i].unsqueeze(
            0) @ all_docs_embeds
        posterior_logits_full = posterior_question_embeddings[i].unsqueeze(
            0) @ all_docs_embeds

        prior_log_dist_full = F.log_softmax(
            prior_logits_full, dim=-1).squeeze()
        posterior_dist_full = F.softmax(
            posterior_logits_full, dim=-1).squeeze()

        KL += F.kl_div(prior_log_dist_full, posterior_dist_full)
    KL /= batch_size
    return KL


def GetPostKL(prior_model_outputs, posterior_model_outputs):
    prior_topk_documents_ids = prior_model_outputs["topk_documents_ids"]
    posterior_topk_documents_ids = posterior_model_outputs["topk_documents_ids"]
    prior_question_embeddings = prior_model_outputs["question_embeddings"]
    posterior_question_embeddings = posterior_model_outputs["question_embeddings"]
    prior_topk_documents_embeddings = prior_model_outputs["topk_documents_embeddings"]
    posterior_topk_documents_embeddings = posterior_model_outputs["topk_documents_embeddings"]

    batch_size = len(prior_topk_documents_ids)
    topk = len(prior_topk_documents_ids[0])

    KL = 0
    for i in range(batch_size):
        all_docs_embeds = posterior_topk_documents_embeddings[i]
        all_docs_embeds = torch.tensor(all_docs_embeds).T.cuda()

        prior_logits_full = prior_question_embeddings[i].unsqueeze(
            0) @ all_docs_embeds
        posterior_logits_full = posterior_question_embeddings[i].unsqueeze(
            0) @ all_docs_embeds

        prior_log_dist_full = F.log_softmax(
            prior_logits_full, dim=-1).squeeze()
        posterior_dist_full = F.softmax(
            posterior_logits_full, dim=-1).squeeze()

        KL += F.kl_div(prior_log_dist_full, posterior_dist_full)
    KL /= batch_size
    return KL


class PriorModel(nn.Module):
    def __init__(self, args, indexed_passages):
        super(PriorModel, self).__init__()
        self.max_length = 512

        if (args.eval_only and args.model_path != None):
            self.config = AutoConfig.from_pretrained(
                os.path.join(args.model_path, "prior", args.checkpoint))
            self.tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(args.model_path, "prior", args.checkpoint))
            self.encoder = DPRQuestionEncoder.from_pretrained(os.path.join(
                args.model_path, "prior", args.checkpoint), config=self.config)
            logger.info("Loading prior model from %s", os.path.join(
                args.model_path, "prior", args.checkpoint))
        else:
            self.config = AutoConfig.from_pretrained(
                args.question_encoder_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.question_encoder_model_name)
            self.encoder = DPRQuestionEncoder.from_pretrained(
                args.question_encoder_model_name, config=self.config)
            logger.info("Loading prior model from %s",
                        args.question_encoder_model_name)

        self.indexed_passages = indexed_passages

    def forward(self, batch, topk):
        # input_ids: batch_size x sequence_length
        input_ids = batch

        # question_embeddings: batch_size x 768
        question_embeddings = self.encoder(
            input_ids=input_ids[:, :self.max_length]).pooler_output
        retrieved_indices = self.indexed_passages.retrieve(
            question_embeddings, topk)

        # topk_documents_embeddings: batch_size x topk x 768
        topk_documents_embeddings = self.indexed_passages.get_field_by_indices(
            retrieved_indices, "embeddings")
        topk_documents_embeddings = torch.tensor(
            topk_documents_embeddings).cuda()

        # logits: batch_size x topk
        logits = torch.bmm(topk_documents_embeddings,
                           question_embeddings.unsqueeze(2)).squeeze(2)

        return logits, torch.tensor(retrieved_indices).cuda(), question_embeddings

    def save_model(self, args, model_name):
        output_dir = os.path.join(args.model_path, "prior", model_name)
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Saving prior model checkpoint to %s", output_dir)
        self.encoder.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        with open(os.path.join(output_dir, "params.json"), "w") as jsonfile:
            json.dump(args.params, jsonfile, indent=4,
                      default=lambda x: str(x))


class PosteriorModel(nn.Module):
    def __init__(self, args, indexed_passages):
        super(PosteriorModel, self).__init__()
        self.max_length = 512

        if (args.eval_only and args.model_path != None):
            self.config = AutoConfig.from_pretrained(
                os.path.join(args.model_path, "posterior", args.checkpoint))
            self.tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(args.model_path, "posterior", args.checkpoint))
            self.encoder = DPRQuestionEncoder.from_pretrained(os.path.join(
                args.model_path, "posterior", args.checkpoint), config=self.config)
            logger.info("Loading posterior model from %s", os.path.join(
                args.model_path, "posterior", args.checkpoint))
        else:
            self.config = AutoConfig.from_pretrained(
                args.question_encoder_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.question_encoder_model_name)
            self.encoder = DPRQuestionEncoder.from_pretrained(
                args.question_encoder_model_name, config=self.config)
            logger.info("Loading posterior model from %s",
                        args.question_encoder_model_name)

        self.indexed_passages = indexed_passages

    def forward(self, batch, topk):
        input_ids, token_type_ids = batch

        # input_ids: batch_size x sequence_length
        # token_type_ids: batch_size x sequence_length
        input_ids = input_ids.cuda()
        token_type_ids = token_type_ids.cuda()

        # question_embeddings: batch_size x 768
        question_embeddings = self.encoder(
            input_ids=input_ids[:, :self.max_length], token_type_ids=token_type_ids[:, :self.max_length]).pooler_output
        retrieved_indices = self.indexed_passages.retrieve(
            question_embeddings, topk)

        # topk_documents_embeddings: batch_size x topk x 768
        topk_documents_embeddings = self.indexed_passages.get_field_by_indices(
            retrieved_indices, "embeddings")
        topk_documents_embeddings = torch.tensor(
            topk_documents_embeddings).cuda()

        # logits: topk x 768
        logits = torch.bmm(topk_documents_embeddings,
                           question_embeddings.unsqueeze(2)).squeeze(2)

        return logits, torch.tensor(retrieved_indices).cuda(), question_embeddings

    def save_model(self, args, model_name):
        output_dir = os.path.join(args.model_path, "posterior", model_name)
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Saving posterior model checkpoint to %s", output_dir)
        self.encoder.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        with open(os.path.join(output_dir, "params.json"), "w") as jsonfile:
            json.dump(args.params, jsonfile, indent=4,
                      default=lambda x: str(x))


class DecoderModel(nn.Module):
    def __init__(self, args):
        super(DecoderModel, self).__init__()

        self.SPECIAL_TOKENS = {
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "additional_special_tokens": ["<speaker1>", "<speaker2>"]
        }
        self.SPECIAL_TOKENS_VALUES = [
            "<bos>", "<eos>", "<pad>", "<speaker1>", "<speaker2>"]
        self.max_length = 512
        self.divide = args.divide

        if (args.eval_only and args.model_path != None):
            self.config = AutoConfig.from_pretrained(
                os.path.join(args.model_path, "decoder", args.checkpoint))
            self.tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(args.model_path, "decoder", args.checkpoint))
            self.decoder = GPT2LMHeadModel.from_pretrained(os.path.join(
                args.model_path, "decoder", args.checkpoint), config=self.config)
            logger.info("Loading decoder model from %s", os.path.join(
                args.model_path, "decoder", args.checkpoint))
        else:
            self.config = AutoConfig.from_pretrained(args.decoder_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.decoder_model_name)
            self.decoder = GPT2LMHeadModel.from_pretrained(
                args.decoder_model_name, config=self.config)

            self.tokenizer.add_special_tokens(self.SPECIAL_TOKENS)
            self.decoder.resize_token_embeddings(len(self.tokenizer))
            logger.info("Loading decoder model from %s",
                        args.decoder_model_name)

        self.bos = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["bos_token"])
        self.eos = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["eos_token"])
        self.pad = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["pad_token"])
        self.speaker1, self.speaker2 = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["additional_special_tokens"])

    def _dec(self, text):
        input_ids = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(text))
        return input_ids

    def _prepare_inputs(self, decoder_input_ids, decoder_response_ids, topk_documents_text, with_eos=True):
        # decoder_input_ids: batch_size x sequence_length
        # decoder_response_ids: batch_size x sequence_length
        # topk_documents_text: batch_size x topk x text_length
        batch_size = len(decoder_input_ids)
        topk = len(topk_documents_text[0])

        list_ids = []
        list_lms = []
        list_type_ids = []
        for i in range(batch_size):
            for j in range(topk):
                knowledge = self._dec(topk_documents_text[i][j])
                history = decoder_input_ids[i]
                response = decoder_response_ids[i]

                sequence = [[self.bos] + knowledge] + [history] + \
                    [response + ([self.eos] if with_eos else [])]
                sequence_with_speaker = [[self.speaker1 if (len(
                    sequence) - i) % 2 == 0 else self.speaker2] + s for i, s in enumerate(sequence[1:])]
                sequence = [sequence[0]] + sequence_with_speaker

                type_ids = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(
                    sequence) for _ in s]
                lms = ([-100] * sum(len(s)
                                    for s in sequence[:-1])) + [-100] + sequence[-1][1:]
                ids = list(chain(*sequence))

                list_ids.append(ids)
                list_lms.append(lms)
                list_type_ids.append(type_ids)

        list_ids = torch.tensor(pad_ids(list_ids, self.pad))
        list_lms = torch.tensor(pad_ids(list_lms, -100))
        list_type_ids = torch.tensor(pad_ids(list_type_ids, self.pad))

        list_ids = list_ids[:, :self.max_length]
        list_lms = list_lms[:, :self.max_length]
        list_type_ids = list_type_ids[:, :self.max_length]

        # decoder_input_ids: batch_size x topk x sequence_length
        # decoder_response_ids: batch_size x topk x sequence_length
        # decoder_token_type_ids: batch_size x topk x sequence_length
        decoder_input_ids = list_ids.reshape(batch_size, topk, -1)
        decoder_response_ids = list_lms.reshape(batch_size, topk, -1)
        decoder_token_type_ids = list_type_ids.reshape(batch_size, topk, -1)

        return decoder_input_ids, decoder_response_ids, decoder_token_type_ids

    def compute_gen_loss_item(self, lm_logits, labels):
        batch_size, topk, sequence_length, _ = lm_logits.shape

        lm_logits = lm_logits.reshape(batch_size * topk, sequence_length, -1)
        labels = labels.reshape(batch_size * topk, -1)

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))

        loss = loss.reshape(batch_size * topk, -1)
        shift_labels = shift_labels.reshape(batch_size * topk, -1)

        loss = loss.sum(dim=-1) / (shift_labels != -100).sum(dim=-1)
        loss = loss.reshape(batch_size, topk)

        return loss

    def forward(self, batch):
        decoder_input_ids, decoder_response_ids = batch

        decoder_input_ids = decoder_input_ids.cuda()
        decoder_response_ids = decoder_response_ids.cuda()

        batch_size, topk, sequence_length = decoder_input_ids.shape
        decoder_input_ids = decoder_input_ids.reshape(batch_size * topk, -1)

        decoder_model_outputs = self.decoder(
            input_ids=decoder_input_ids, token_type_ids=None)

        lm_logits = decoder_model_outputs[0]
        lm_logits = lm_logits.reshape(batch_size, topk, sequence_length, -1)
        # FIXME divide parameter
        lm_logits = lm_logits / self.divide

        loss = self.compute_gen_loss_item(lm_logits, decoder_response_ids)

        return loss, lm_logits

    # NOTE only works with batch size 1
    def generate_from_1_doc(self, args, decoder_input_ids, best_document_decoder_text, return_ids=False):
        decoder_input_ids_, _, _ = self._prepare_inputs(
            [decoder_input_ids], [[]], [[best_document_decoder_text]], with_eos=False)
        decoder_input_ids_ = decoder_input_ids_.squeeze(1).cuda()

        output = self.decoder.generate(
            input_ids=decoder_input_ids_,
            max_length=args.generation_args["max_length"] +
            decoder_input_ids_.shape[1],
            min_length=args.generation_args["min_length"],
            top_k=args.generation_args["top_k"],
            top_p=args.generation_args["top_p"],
            temperature=args.generation_args["temperature"],
            bos_token_id=self.bos,
            eos_token_id=self.eos,
            pad_token_id=self.pad
        )

        output = output[0][decoder_input_ids_.shape[1]:]
        output_text = self.tokenizer.decode(
            output, skip_special_tokens=True)

        if (return_ids):
            return output_text, output.cpu().numpy().tolist()
        return output_text

    # NOTE only works with batch size 1
    def generate_from_k_docs(self, args, decoder_input_ids, topk_documents_decoder_text, prior_dist):
        p_y_given_zx = []
        output_text = None
        p_max = 0
        for i in range(len(prior_dist)):
            text_, decoder_response_ids = self.generate_from_1_doc(
                args, decoder_input_ids, topk_documents_decoder_text[i], return_ids=True)

            decoder_input_ids_, decoder_response_ids_, _ = self._prepare_inputs(
                [decoder_input_ids], [decoder_response_ids], [[topk_documents_decoder_text[i]]], with_eos=False)
            decoder_loss, _ = self([decoder_input_ids_, decoder_response_ids_])

            p_y_given_zx = torch.exp(-decoder_loss).squeeze().cpu().numpy()
            p_y_given_x = p_y_given_zx * prior_dist[i]
            if (p_y_given_x > p_max):
                p_max = p_y_given_x
                output_text = text_

        return output_text

    def save_model(self, args, model_name):
        output_dir = os.path.join(args.model_path, "decoder", model_name)
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Saving decoder model checkpoint to %s", output_dir)
        self.decoder.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        with open(os.path.join(output_dir, "params.json"), "w") as jsonfile:
            json.dump(args.params, jsonfile, indent=4,
                      default=lambda x: str(x))


class UnsupervisedModel(nn.Module):
    def __init__(self, args, indexed_passages):
        super(UnsupervisedModel, self).__init__()

        self.modeling_method = args.modeling_method
        if (self.modeling_method == "VRAG"):
            self.kl_beta = args.kl_beta
        elif (self.modeling_method == "RL"):
            self.num_samples = args.num_samples

        self.topk = args.topk
        self.prior_model = PriorModel(args, indexed_passages)
        self.posterior_model = PosteriorModel(args, indexed_passages)
        self.decoder_model = DecoderModel(args)

        if (args.n_gpus > 1):
            self.prior_model = nn.DataParallel(self.prior_model)
            self.posterior_model = nn.DataParallel(self.posterior_model)
            self.decoder_model = nn.DataParallel(self.decoder_model)
            self.parallel = True
        else:
            self.parallel = False

    def SampleCategorical(self, dist, num_samples):
        samples = []
        for _ in range(num_samples):
            s = torch.distributions.categorical.Categorical(
                dist).sample().unsqueeze(1)
            samples.append(s)
        samples = torch.cat(samples, dim=1)
        return samples

    def SelectByIndices(self, x, indices):
        l = []
        for i in range(indices.shape[0]):
            l.append([])
            for j in range(indices.shape[1]):
                l[-1].append(x[i][indices[i][j]])
        l = torch.tensor(l)
        return l

    def forward(self, batch):
        (prior_input_ids,
         posterior_input_ids,
         posterior_token_type_ids,
         decoder_input_ids,
         decoder_response_ids,
         doc_ids,
         q_ids) = batch

        if (self.modeling_method == "RAG"):
            prior_logits, prior_indices, _ = self.prior_model(
                prior_input_ids.cuda(), self.topk)
            p_z_given_x = F.softmax(prior_logits, dim=-1) + EPSILON

            prior_indices = prior_indices.cpu().tolist()

            if (self.parallel):
                prior_topk_documents_text = self.prior_model.module.indexed_passages.get_field_by_indices(
                    prior_indices, "text")
                prior_topk_documents_ids = self.prior_model.module.indexed_passages.get_field_by_indices(
                    prior_indices, "id")

                decoder_input_ids_, decoder_response_ids_, _ = self.decoder_model.module._prepare_inputs(
                    decoder_input_ids, decoder_response_ids, prior_topk_documents_text)
            else:
                prior_topk_documents_text = self.prior_model.indexed_passages.get_field_by_indices(
                    prior_indices, "text")
                prior_topk_documents_ids = self.prior_model.indexed_passages.get_field_by_indices(
                    prior_indices, "id")

                decoder_input_ids_, decoder_response_ids_, _ = self.decoder_model._prepare_inputs(
                    decoder_input_ids, decoder_response_ids, prior_topk_documents_text)

            decoder_loss, _ = self.decoder_model(
                [decoder_input_ids_.cuda(), decoder_response_ids_.cuda()])

            p_y_given_zx = torch.exp(-decoder_loss) + EPSILON

            p_y_given_x = (p_z_given_x * p_y_given_zx).sum(dim=-1) + EPSILON
            loss = (-torch.log(p_y_given_x)).mean()

            print("loss =", loss)
            print("p(z|x) =", p_z_given_x)
            print("p(y|zx) =", p_y_given_zx)
            print("p(y|x) =", p_y_given_x)
            print("topk_documents_ids =", prior_topk_documents_ids)
            print("doc_ids =", doc_ids)
            print("q_ids =", q_ids)
            print()

            return loss
        elif (self.modeling_method == "VRAG"):
            posterior_logits, posterior_indices, posterior_question_embeddings = self.posterior_model(
                [posterior_input_ids.cuda(), posterior_token_type_ids.cuda()], self.topk)

            prior_logits, prior_indices, prior_question_embeddings = self.prior_model(
                prior_input_ids.cuda(), self.topk)

            posterior_indices = posterior_indices.cpu().tolist()
            prior_indices = prior_indices.cpu().tolist()

            if (self.parallel):
                posterior_topk_documents_text = self.posterior_model.module.indexed_passages.get_field_by_indices(
                    posterior_indices, "text")
                posterior_topk_documents_ids = self.posterior_model.module.indexed_passages.get_field_by_indices(
                    posterior_indices, "id")
                posterior_topk_documents_embeddings = self.posterior_model.module.indexed_passages.get_field_by_indices(
                    posterior_indices, "embeddings")

                prior_topk_documents_ids = self.prior_model.module.indexed_passages.get_field_by_indices(
                    prior_indices, "id")
                prior_topk_documents_embeddings = self.prior_model.module.indexed_passages.get_field_by_indices(
                    prior_indices, "embeddings")

                decoder_input_ids_, decoder_response_ids_, _ = self.decoder_model.module._prepare_inputs(
                    decoder_input_ids, decoder_response_ids, posterior_topk_documents_text)
            else:
                posterior_topk_documents_text = self.posterior_model.indexed_passages.get_field_by_indices(
                    posterior_indices, "text")
                posterior_topk_documents_ids = self.posterior_model.indexed_passages.get_field_by_indices(
                    posterior_indices, "id")
                posterior_topk_documents_embeddings = self.posterior_model.indexed_passages.get_field_by_indices(
                    posterior_indices, "embeddings")

                prior_topk_documents_ids = self.prior_model.indexed_passages.get_field_by_indices(
                    prior_indices, "id")
                prior_topk_documents_embeddings = self.prior_model.indexed_passages.get_field_by_indices(
                    prior_indices, "embeddings")

                decoder_input_ids_, decoder_response_ids_, _ = self.decoder_model._prepare_inputs(
                    decoder_input_ids, decoder_response_ids, posterior_topk_documents_text)

            decoder_loss, _ = self.decoder_model(
                [decoder_input_ids_, decoder_response_ids_])

            posterior_dist = F.softmax(
                posterior_logits, dim=-1) + EPSILON

            e = (posterior_dist * decoder_loss).sum(dim=-1)
            loss = e.mean()

            prior_model_outputs = {
                "topk_documents_ids": prior_topk_documents_ids,
                "question_embeddings": prior_question_embeddings,
                "topk_documents_embeddings": prior_topk_documents_embeddings
            }
            posterior_model_outputs = {
                "topk_documents_ids": posterior_topk_documents_ids,
                "question_embeddings": posterior_question_embeddings,
                "topk_documents_embeddings": posterior_topk_documents_embeddings
            }

            KL = GetUnionKL(prior_model_outputs, posterior_model_outputs)
            loss += self.kl_beta * KL

            print("loss =", loss)
            print("KL =", KL)
            print("decoder_loss =", decoder_loss)
            print("posterior_dist =", posterior_dist)
            print("e =", e)
            print("topk_documents_ids =",
                  posterior_model_outputs["topk_documents_ids"])
            print("doc_ids =", doc_ids)
            print("q_ids =", q_ids)
            print()

            return loss
        elif (self.modeling_method == "RL"):
            prior_logits, prior_indices, prior_question_embeddings = self.prior_model(
                prior_input_ids.cuda(), self.topk)

            prior_indices = prior_indices.cpu().tolist()
            # prior_dist: batch_size x topk
            prior_dist = F.softmax(prior_logits, dim=-1) + EPSILON

            # NOTE tmp is not the same as prior_indices
            tmp = self.SampleCategorical(prior_dist, self.num_samples)

            # prior_sampled_indices: batch_size x num_samples
            prior_sampled_indices = self.SelectByIndices(prior_indices, tmp)
            # prior_sampled_dist: batch_size x num_samples
            prior_sampled_dist = self.SelectByIndices(prior_dist, tmp)

            if (self.parallel):
                prior_sampled_documents_text = self.prior_model.module.indexed_passages.get_field_by_indices(
                    prior_sampled_indices, "text")
                prior_sampled_documents_ids = self.prior_model.module.indexed_passages.get_field_by_indices(
                    prior_sampled_indices, "id")

                decoder_input_ids_, decoder_response_ids_, _ = self.decoder_model.module._prepare_inputs(
                    decoder_input_ids, decoder_response_ids, prior_sampled_documents_text)
            else:
                prior_sampled_documents_text = self.prior_model.module.indexed_passages.get_field_by_indices(
                    prior_sampled_indices, "text")
                prior_sampled_documents_text = self.prior_model.indexed_passages.get_field_by_indices(
                    prior_sampled_indices, "id")

                decoder_input_ids_, decoder_response_ids_, _ = self.decoder_model._prepare_inputs(
                    decoder_input_ids, decoder_response_ids, prior_sampled_documents_text)

            # decoder_loss: batch_size x num_samples
            decoder_loss, _ = self.decoder_model(
                [decoder_input_ids_, decoder_response_ids_])

            loss = decoder_loss + decoder_loss.detach() * torch.log(prior_sampled_dist)
            loss = e.mean()

            print("loss =", loss)
            print("decoder_loss =", decoder_loss)
            print("prior_sampled_dist =", prior_sampled_dist)
            print("prior_sampled_documents_ids =", prior_sampled_documents_ids)
            print("doc_ids =", doc_ids)
            print("q_ids =", q_ids)
            print()

            return loss

    def save_model(self, args, model_name):
        if (self.parallel):
            self.prior_model.module.save_model(args, model_name)
            self.posterior_model.module.save_model(args, model_name)
            self.decoder_model.module.save_model(args, model_name)
        else:
            self.prior_model.save_model(args, model_name)
            self.posterior_model.save_model(args, model_name)
            self.decoder_model.save_model(args, model_name)

    def GetParameters(self):
        params = list(self.prior_model.parameters()) + \
            list(self.decoder_model.parameters())
        if (self.modeling_method == "VRAG"):
            params += list(self.posterior_model.parameters())
        return params
