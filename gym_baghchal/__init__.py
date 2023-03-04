from gym.envs.registration import register

# register(
#     id='Baghchal-v0',
#     entry_point='baghchal_env:BaghchalEnv',
#     kwargs={'board_size': 5, 'num_tigers': 4, 'num_goats': 20},
# )



register(
     id="gym_baghchal/Baghchal-v0",
     entry_point="gym_baghchal.envs:BaghchalEnv",
     kwargs={'board_size': 5, 'num_tigers': 4, 'num_goats': 20},
     max_episode_steps=300,
)