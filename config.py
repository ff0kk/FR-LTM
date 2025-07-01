import json
import copy

class FR_LTM_Config(object):
    """Configuration class to store the configuration of a `FR_LTM_Model`.
    """
    def __init__(self,
                 grid_num=-1,
                 embedding_size=768,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=1536,
                 type_grid_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 rotary_value=False,
                 is_decoder=False,
                 add_cross_attention=False,
                 chunk_size_feed_forward=None,
                 use_cache=False,
                 num_labels=-1,
                 pad_token_id=0,
                 scaling_factor=1,
                 use_unidirectional_selfAttention4downstream=False,
                 use_mean4staticEmb=False,
                 activation_checkpointing=False,
                 activation_checkpoint_interval=1,
                 n_multi_token=0):
        """Constructs FR_LTM_Config.

        Args:
            grid_num: Number of grid of `inputs_ids` in `FR_LTM_Model`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_grid_size: The size of the `grid_type_ids` passed into
                `FR_LTM_Model`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.grid_num = grid_num
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_grid_size = type_grid_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.rotary_value = rotary_value
        self.is_decoder = is_decoder
        self.add_cross_attention = add_cross_attention
        if chunk_size_feed_forward is None:
            self.chunk_size_feed_forward = 0
        else:
            self.chunk_size_feed_forward = chunk_size_feed_forward
        self.use_cache = use_cache
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id
        self.scaling_factor = scaling_factor
        self.use_unidirectional_selfAttention4downstream = use_unidirectional_selfAttention4downstream
        self.use_mean4staticEmb = use_mean4staticEmb
        self.activation_checkpointing = activation_checkpointing
        self.activation_checkpoint_interval = activation_checkpoint_interval
        self.n_multi_token = n_multi_token
        
    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `FR_LTM_Config` from a Python dictionary of parameters."""
        config = FR_LTM_Config()
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `FR_LTM_Config` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())