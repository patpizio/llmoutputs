import torch
import numpy as np, transformers as trf
from plotly.subplots import make_subplots
from warnings import warn
import plotly.graph_objects as go

class LLMOutput:
    def __init__(self, inputs, outputs, tokenizer):
        self.input_ids = inputs['input_ids'].cpu()
        self.sequences = outputs['sequences'].flatten().cpu().numpy()
        self.sequences_as_tensor = outputs['sequences'].cpu()
        self.scores = tuple([t.cpu() for t in outputs['scores']])
        self.tokenizer = tokenizer
        self.len_input = len(self.input_ids.flatten())
        self.len_total = len(self.sequences)
        self.len_output = len(self.sequences) - self.len_input
        self.generated_text = tokenizer.decode(self.sequences)
        # Set the start-of-string character for the chosen model
        if isinstance(self.tokenizer, (trf.models.llama.tokenization_llama_fast.LlamaTokenizerFast,
                                       trf.models.t5.tokenization_t5_fast.T5TokenizerFast,
                                       trf.models.llama.tokenization_llama_fast.LlamaTokenizer,
                                       trf.models.t5.tokenization_t5_fast.T5TokenizerFast,
                                      )):
            self.SOS_character = '▁'  # LLama and T5 (SentencePiece)
        else:
            self.SOS_character = 'Ġ'  # Falcon
    
    def get_logits(self, normalized=False):
        "# Return a 2D tensor, (n. ouput tokens) x (n. tokens in vocabulary)"
        tensor_3d = torch.stack(self.scores, axis=0)
        tensor_2d = tensor_3d.view(tensor_3d.shape[0], -1)
        if normalized:
            return tensor_2d/tensor_2d.max()
        else:
            return tensor_2d
    
    def get_probabilities(self, temperature=1.0):
        # 'scores' for a k-token output is a tuple of k 2D-tensors that only use 1 dimension. 
        tensor_2d = self.get_logits() / temperature
        return torch.nn.Softmax(dim=1)(tensor_2d)
    
    def token_logits(self, token, normalized=False, add_SOS_character=True):
        """NB: If a token is out of vocabulary, it will get the score for the token '<unk>'
        check if it works the same for models other than Flan-T5
        """
        if add_SOS_character:
            token = f'{self.SOS_character}{token}'
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        return [float(self.get_logits(normalized)[step][token_id]) for step in range(self.len_output)]

    def token_proba(self, token, temperature=1.0, add_SOS_character=True):
        """NB: If a token is out of vocabulary, it will get the score for the token '<unk>'
        check if it works the same for models other than Flan-T5
        """
        if add_SOS_character:
            token = f'{self.SOS_character}{token}'
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        return [float(self.get_probabilities(temperature)[step][token_id]) for step in range(self.len_output)]
        
    def top_token_ids(self, threshold=-np.inf):
        "Return the index of the tokens whose score exceeds a threshold, for each output step"
        indexes = []
        for tensor in self.scores:
            candidates = np.argwhere(tensor.flatten().cpu() > threshold).numpy()[0]
            ordering_mask = np.argsort(tensor[0][candidates].cpu())
            candidates = candidates[ordering_mask]
            if not isinstance(candidates, np.ndarray):
                indexes.append(np.array([candidates]))
            else:
                indexes.append(candidates)
        return indexes
    
    def plot_token_scores(self, min_score=-np.inf, softmax=False, normalized=False, temperature=1.0, width=600):
        if softmax and normalized:
            warn("Note that normalization is not applied when using softmax.")
        top_ids = self.top_token_ids(threshold=min_score)
        fig = make_subplots(rows=len(top_ids), cols=1)
        for step, candidates in enumerate(top_ids):  
            if softmax:
                x_axis = self.get_probabilities(temperature)[step][candidates]
            else:
                x_axis = self.get_logits(normalized)[step][candidates]
            fig.append_trace(
                go.Bar(
                    y=self.tokenizer.convert_ids_to_tokens(candidates), 
                    x=x_axis,
                    orientation='h'
                ),
                row=step+1, col=1
            )
        fig.update_layout(
            width=500, 
            height=400*len(top_ids),
            showlegend=False,
            xaxis_title = 'probability' if softmax else 'logit'
        )
        return fig  
    
  
        
    def print_logits(self, model, normalized=True):
        # Does not work as it needs 'sequences' to have 
        transition_scores = model.compute_transition_scores(
            self.sequences_as_tensor, self.scores, normalize_logits=normalized
        )
        input_length = self.input_ids.shape[1]
        generated_tokens = self.sequences_as_tensor[:, input_length:]
        for tok, score in zip(generated_tokens[0], transition_scores[0]):
            print(f"| {tok.cpu():5d} | {self.tokenizer.decode(tok.cpu()):8s}\t| {score.cpu().numpy():.4f} | {np.exp(score.cpu().numpy()):.2%}")          