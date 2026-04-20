import gfootball.env as football_env
import json
import os

# Create the environment
env = football_env.create_environment(
    env_name='11_vs_11_easy_stochastic', 
    representation='raw', 
    render=True 
)

# This is the secret sauce: 
# Access the raw engine that always returns 4 values, ignoring gym's wrappers
raw_env = env.unwrapped

obs = env.reset()
pass_count = 0
last_owner = -1
all_frames_coordinates = []

print("Game Started! Recording coordinates until 11 passes are made...")

try:
    while pass_count < 11:
        # 1. Step the raw environment (returns 4 values)
        obs_raw, reward, done, info = raw_env.step([0])

        # 2. Get the "pretty" dictionary observation directly from the environment
        # This bypasses the need to translate the raw numpy array manually
        obs = env.unwrapped.observation()

        # 3. Handle the case where obs might be a list (multi-agent style)
        if isinstance(obs, list):
            obs = obs[0]

        # Record the data
        frame_snapshot = {
            "step": len(all_frames_coordinates),
            "ball": obs['ball'].tolist(),
            "home_team": obs['left_team'].tolist(),
            "away_team": obs['right_team'].tolist(),
        }
        all_frames_coordinates.append(frame_snapshot)

        # Pass Detection Logic
        current_owner = obs['ball_owned_player']
        current_team = obs['ball_owned_team']

        if current_team == 0:
            if last_owner != -1 and current_owner != last_owner:
                pass_count += 1
                print(f"Pass {pass_count}/11 detected! ({last_owner} -> {current_owner})")
            last_owner = current_owner
        elif current_team == 1:
            last_owner = -1

        if done:
            env.reset()
            last_owner = -1

    # Save the data
    with open('pass_data_11.json', 'w') as f:
        json.dump(all_frames_coordinates, f)
    
    print(f"\nSuccess! Captured {len(all_frames_coordinates)} frames.")

finally:
    env.close()