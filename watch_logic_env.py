import time
import os
from benchmarl.environments.logic_env.factory import SynchronizedFactory

def main():
    print("Inicjalizacja środowiska...")
    env = SynchronizedFactory()
    obs, info = env.reset()
    
    # Wyczyść ekran
    os.system('cls' if os.name == 'nt' else 'clear')
    env.render()
    time.sleep(1)
    
    done = False
    step = 0
    
    while not done and step < 100:
        # Losowe akcje dla każdego agenta
        actions = {
            agent: env.action_spaces[agent].sample() 
            for agent in env.agents
        }
        
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # Wyczyść ekran i wyrysuj nową klatkę
        os.system('cls' if os.name == 'nt' else 'clear')
        env.render()
        
        print(f"\nKrok: {step}")
        print(f"Akcje: {actions}")
        print(f"Nagrody: {rewards}")
        
        done = any(terminations.values()) or any(truncations.values())
        step += 1
        
        # Poczekaj chwilę, żeby animacja była widoczna (0.2 sekundy)
        time.sleep(0.2)
        
    print("\nZakończono epizod!")

if __name__ == "__main__":
    main()
