from rnn import RNN_LSTM, RNN_GRU
from Transformer import Encoder
from fcn import FCN


def load_model(args):
    model_type = args.type.lower()

    if model_type in ['lstm', 'gru']:
        model_class = RNN_LSTM if model_type == 'lstm' else RNN_GRU
        model = model_class(
            feature_channel=args.feature_channel,
            output_channel=args.output_channel,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers
        )
    
    elif model_type == 'transformer':
        model = Encoder(
            feature_channel=args.feature_channel,
            output_channel=args.output_channel,
            embed_size=args.embed_size,
            num_layers=args.num_layers,
            heads=args.nhead,
            forward_expansion=args.forward_expansion if args.forward_expansion is not None else 1,
            seq_length=args.seq_length,
            dropout=args.dropout
        )
    
    elif model_type in ['fcn', 'fullyconnected']:
        model = FCN(
            feature_channel=args.feature_channel,
            output_channel=args.output_channel,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            seq_length=args.seq_length,
            dim_expand=0
        )
    else:
        raise ValueError(f"Model type '{args.type}' is not implemented.")

    return model
