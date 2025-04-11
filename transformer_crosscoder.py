import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


#############################
# Step 1: Define a Transformer-based Feature Extractor
#############################

# This extractor embeds the CartPole observation (a 4-dimensional vector) as a sequence
# (each element becomes a “token”) and passes it through a simple transformer encoder.
class TransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64, d_model=32, nhead=4, num_layers=1):
        super(TransformerExtractor, self).__init__(observation_space, features_dim)
        self.obs_dim = observation_space.shape[0]  # CartPole has 4 features
        # Embed each scalar into a d_model-dimensional space.
        self.embed = nn.Linear(1, d_model)
        # A very simple transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Pool and map down to our chosen feature dimension.
        self.fc = nn.Linear(d_model, features_dim)

    def forward(self, observations):
        # observations is (batch, 4); view it as a sequence of 4 tokens of dim 1.
        batch_size = observations.shape[0]
        x = observations.view(batch_size, self.obs_dim, 1)
        x = self.embed(x)  # (batch, 4, d_model)
        x = x.transpose(0, 1)  # (4, batch, d_model) for transformer input
        x = self.transformer_encoder(x)  # (4, batch, d_model)
        x = x.mean(dim=0)  # (batch, d_model): average pooling over tokens
        features = self.fc(x)  # (batch, features_dim)
        return features


# Specify our custom extractor for the policy.
policy_kwargs = dict(
    features_extractor_class=TransformerExtractor,
    features_extractor_kwargs=dict(features_dim=64, d_model=32, nhead=4, num_layers=1)
)

#############################
# Step 2: Train a PPO Agent on CartPole Using the Transformer Architecture
#############################

env = gym.make('CartPole-v1')
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

# Train for a short period for demonstration purposes.
model.learn(total_timesteps=10000)
model.save("ppo_cartpole_transformer")


#############################
# Step 3: Train a "CrossCoder" on the Transformer's Representations
#############################

# Define a dummy CrossCoder network that maps transformer features to the policy’s logits.
class CrossCoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CrossCoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Helper: extract transformer features from the policy's custom extractor.
def get_transformer_features(model, observations):
    with torch.no_grad():
        obs_tensor = torch.tensor(observations, dtype=torch.float32)
        features = model.policy.features_extractor(obs_tensor)
    return features


# Create a dataset by running the environment and recording both features and policy logits.
def create_dataset(model, env, num_samples=1000):
    features_list, logits_list = [], []
    obs = env.reset()
    for _ in range(num_samples):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        # Extract features using the transformer's extractor.
        features = model.policy.features_extractor(obs_tensor)
        # Get policy logits from the PPO policy network.
        logits = model.policy.mlp_extractor.policy_net(features)
        features_list.append(features.squeeze(0).numpy())
        logits_list.append(logits.squeeze(0).numpy())

        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    return np.array(features_list), np.array(logits_list)


# Build the dataset.
X, Y = create_dataset(model, env, num_samples=1000)
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

input_dim = X_tensor.shape[1]
output_dim = Y_tensor.shape[1]
crosscoder = CrossCoder(input_dim, output_dim)

optimizer = optim.Adam(crosscoder.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Train the CrossCoder to mimic the policy’s output given the transformer features.
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = crosscoder(X_tensor)
    loss = criterion(outputs, Y_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss {loss.item()}")


#############################
# Step 4: "Extract" and Hard-Code the Algorithm
#############################

# In our toy example, we simulate extraction by wrapping the trained transformer extractor
# and the crosscoder into a hard-coded policy. In practice, one might inspect or modify the network's
# weights to obtain an explicit algorithm.
class HardCodedPolicy:
    def __init__(self, features_extractor, crosscoder):
        self.features_extractor = features_extractor  # from the trained PPO model
        self.crosscoder = crosscoder
        self.crosscoder.eval()

    def act(self, obs):
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            features = self.features_extractor(obs_tensor)
            logits = self.crosscoder(features)
            probs = F.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1).item()
        return action


# Instantiate the hard-coded policy.
hardcoded_policy = HardCodedPolicy(model.policy.features_extractor, crosscoder)

# Test the new policy in the environment.
obs = env.reset()
total_reward = 0
for _ in range(200):
    action = hardcoded_policy.act(obs)
    obs, reward, done, _ = env.step(action)
    total_reward += reward
    if done:
        break
print("Total reward using hard-coded policy:", total_reward)
