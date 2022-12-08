import torch
import torch.nn as nn
from .utils import SupermaskLinear

"""
Parameters for SNN
"""

HIDDEN_DIM = 64
NEURON_VTH = 0.5
NEURON_VDECAY = 0.75
SPIKE_TS = 4

class SPN_Connections(nn.Module):

    def __init__(self, input_space_dim, action_space_dim):
   
        super().__init__()  # 学习结构的话 bias=False 效果更好，也更合理。
        self.in_dim = input_space_dim
        self.out_dim = action_space_dim

        self.hidden_layer = SupermaskLinear(input_space_dim, HIDDEN_DIM)

        self.out_layer = SupermaskLinear(HIDDEN_DIM, action_space_dim)


    def neuron_model_LIF(self, syn_func, pre_layer_output, volt, spike):
        """
        LIF Neuron Model
        :param syn_func: synaptic function
        :param pre_layer_output: output from pre-synaptic layer
        :param volt: voltage of last step
        :param spike: spike of last step
        :return: volt, spike
        """
        volt = volt * NEURON_VDECAY * (1. - spike) + syn_func(pre_layer_output)
        spike = volt.gt(NEURON_VTH).float()
        return volt, spike

    def neuron_model_LI(self, syn_func, pre_layer_output,volt):
        """
        LI Neuron Model
        :param syn_func: synaptic function
        :param pre_layer_output: output from pre-synaptic layer
        :param volt: voltage of last step
        :return: volt
        """
        volt = volt * NEURON_VDECAY + syn_func(pre_layer_output)

        return volt

    def forward(self, obs):
        obs = torch.as_tensor(obs).float().detach()
        batch_size = obs.shape[0]

        # Define LIF Neuron states: Voltage and Spike
        hidden_states = [torch.zeros(batch_size, HIDDEN_DIM) for _ in range(2)]

        out_states = torch.zeros(batch_size, self.out_dim)
        out_act = []

        # Start Spike Timestep Iteration
        for step in range(SPIKE_TS):

            in_spike_t = obs

            hidden_states[0], hidden_states[1] = self.neuron_model_LIF(
                self.hidden_layer, in_spike_t,
                hidden_states[0], hidden_states[1]
            )

            out_states= self.neuron_model_LI(
                self.out_layer, hidden_states[1],out_states
            )
            out_act.append(out_states)

        out = torch.cat(out_act,dim=0)
        out_final,_ = torch.max(out,dim=0)

        return out_final

    def reset_state(self):
        pass

    def get_weights(self):
        return nn.utils.parameters_to_vector(self.parameters()).detach().numpy()

    def load_weights(self,parameter_vector):
        return nn.utils.vector_to_parameters(torch.tensor(parameter_vector, dtype=torch.float32), self.parameters())