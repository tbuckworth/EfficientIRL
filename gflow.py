from imitation.rewards.reward_nets import BasicShapedRewardNet, BasicRewardNet, ShapedRewardNet
from stable_baselines3.common.policies import ActorCriticPolicy

from helper_local import get_policy_for


class GFLOW:
    def __init__(self,
                 observation_space,
                 action_space,
                 demonstrations,
                 gamma=0.98,
                 batch_size=256,
                 l2_weight=0.001,
                 rng=None,
                 custom_logger=None,
                 net_arch=None,
                 use_state=True,
                 use_action=False,
                 use_next_state=False,
                 ):
        if net_arch is None:
            net_arch = [256, 256, 256, 256]
        self.observation_space = observation_space
        self.action_space = action_space
        self.demonstrations = demonstrations
        self.gamma = gamma
        self.batch_size = batch_size
        self.l2_weight = l2_weight
        self.rng = rng
        self.custom_logger = custom_logger

        self.forward_policy = get_policy_for(observation_space, action_space, net_arch)
        self.backward_policy = get_policy_for(observation_space, action_space, net_arch)

        value_func = lambda obs: self.forward_policy.predict_values(obs).squeeze()

        self.output_reward_function = BasicRewardNet(
            observation_space,
            action_space,
            use_state,
            use_action,
            use_next_state
        )
        self.reward_function = ShapedRewardNet(self.output_reward_function,
                                               value_func,
                                               gamma)
