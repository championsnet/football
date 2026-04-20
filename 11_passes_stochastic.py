import gfootball.env as football_env
import json

# --- THE SETUP ---
env = football_env.create_environment(
    env_name='11_vs_11_hard_stochastic', 
    representation='raw', 
    render=True,
    other_config_options={
        'action_set': 'full',
        'players': ['bot:left_players=1', 'bot:right_players=1']
    }
)

raw_env = env.unwrapped
obs = env.reset()

pass_count = 0
last_teammate_owner = -1  # Tracks the last player from Team 0 who had it
all_frames_coordinates = []

print("Game Started! Counting 11 successful Home Team passes...")

try:
    while pass_count < 11:
        # Step the engine
        raw_env.step([])
        
        # Get dictionary observation
        obs = raw_env.observation()
        if isinstance(obs, list):
            obs = obs[0]

        # Record movement data
        all_frames_coordinates.append({
            "step": len(all_frames_coordinates),
            "ball": obs['ball'].tolist(),
            "home_team": obs['left_team'].tolist(),
            "away_team": obs['right_team'].tolist(),
        })

        current_owner = obs['ball_owned_player']
        current_team = obs['ball_owned_team']

        # LOGIC:
        # 1. If Team 0 (Home) currently has the ball
        if current_team == 0:
            # If someone different just caught it, and we know who passed it
            if last_teammate_owner != -1 and current_owner != last_teammate_owner:
                pass_count += 1
                print(f"COMPLETE PASS {pass_count}/11: Player {last_teammate_owner} -> Player {current_owner}")
            
            # Update the last known teammate to touch the ball
            last_teammate_owner = current_owner
            
        # 2. If Team 1 (Away) gets the ball, the passing chain is BROKEN
        elif current_team == 1:
            if last_teammate_owner != -1:
                print("Possession lost! Passing chain reset.")
            last_teammate_owner = -1

        # 3. If current_team is -1 (ball in air/loose), we keep last_teammate_owner 
        # so we can detect who catches it next.

        if obs['steps_left'] == 0:
            env.reset()
            last_teammate_owner = -1

    with open('pass_data_11.json', 'w') as f:
        json.dump(all_frames_coordinates, f)
    
    print(f"\nMission Accomplished! 11 successful passes saved.")

finally:
    env.close()