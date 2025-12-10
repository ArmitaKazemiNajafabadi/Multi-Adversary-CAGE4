from gymnasium.spaces import Dict
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

class TorchActionMaskModelHppo(TorchModelV2, nn.Module):
    """PyTorch version of above TorchActionMaskModel."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observations" in orig_space.spaces
        )

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)
        
        self.internal_model = TorchFC(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )
        
    def forward(self, input_dict, state, seq_lens):
        '''
        action[b, a] == 1 -> action a is valid in batch_b
        action[b, a] == 0 -> action a is not valid
        '''
        action_mask = input_dict["obs"]["action_mask"]
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})
        '''
        log(1) == 0 for valid actions
        log(0) == -inf for invalid actions
        torch.clamp() -> if -inf then take a very large neg. number
        '''
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        # For an invalid state perform logits - inf approx -inf
        masked_logits = logits + inf_mask
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()