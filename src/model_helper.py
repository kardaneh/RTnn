import torch


class ModelUtils:
    """
    Utility class for model inspection, checkpointing, and memory profiling.
    """

    def __init__(self):
        pass

    @staticmethod
    def get_parameter_number(model):
        """
        Returns the total and trainable number of parameters in a model.

        Args:
            model (torch.nn.Module): The model to inspect.

        Returns:
            dict: Dictionary with total and trainable parameter counts.
        """
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {"Total": total_num, "Trainable": trainable_num}

    @staticmethod
    def get_memory_usage(model, input_shape):
        """
        Prints the memory usage statistics for a model given an input shape.

        Args:
            model (torch.nn.Module): The model to profile.
            input_shape (tuple): Input shape, excluding batch size.
        """
        from torchstat import stat

        stat(model, input_shape)

    @staticmethod
    def print_model_layer(model):
        """
        Prints model parameter names along with their gradient requirement.
        """
        for name, param in model.named_parameters():
            print(f"name: {name},\t grad: {param.requires_grad}")

    @staticmethod
    def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
        """
        Saves model and optimizer state to a file.

        Args:
            state (dict): Dictionary containing model and optimizer state_dicts.
            filename (str): File path to save the checkpoint.
        """
        print("=> Saving checkpoint")
        torch.save(state, filename)

    @staticmethod
    def load_checkpoint(checkpoint, model, optimizer):
        """
        Loads model and optimizer state from a checkpoint.

        Args:
            checkpoint (dict): Loaded checkpoint dictionary.
            model (torch.nn.Module): Model to load weights into.
            optimizer (torch.optim.Optimizer): Optimizer to restore state.
        """
        print("=> Loading checkpoint")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
