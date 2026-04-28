import time
import os
from benchmarl.environments.logic_env.factory import SynchronizedFactory


def main():
    print("Initializing environment...")
    env = SynchronizedFactory()
    obs, info = env.reset()
    
    # Clear the screen
    os.system('cls' if os.name == 'nt' else 'clear')
    env.render()
    time.sleep(1)
    
    done = False
    step = 0
    
    while not done and step < 100:
        # Sample random actions for each agent
        actions = {
            agent: env.action_spaces[agent].sample() 
            for agent in env.agents
        }
        
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # Clear screen and render new frame
        os.system('cls' if os.name == 'nt' else 'clear')
        env.render()
        
        print(f"\nStep: {step}")
        print(f"Actions: {actions}")
        print(f"Rewards: {rewards}")
        
        done = any(terminations.values()) or any(truncations.values())
        step += 1
        
        # Wait a moment so the animation is visible (0.2 seconds)
        time.sleep(0.2)
        
    print("\nEpisode finished!")


if __name__ == "__main__":
    main()