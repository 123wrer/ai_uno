import rlcard
from rlcard.agents import DQNAgent
from rlcard.utils import get_device
import torch
import os

def print_state(state):
    raw_obs = state['raw_obs']
    print("\n" + "="*40)
    print(f"🃏 Карта на столе: {raw_obs['target']}")
    
    # Немного красоты для вывода карт (перевод на русский)
    hand = [card.replace('r', 'Красный').replace('b', 'Синий').replace('g', 'Зеленый').replace('y', 'Желтый') for card in raw_obs['hand']]
    print(f"🖐 Ваша рука: {', '.join(hand)}")
    print("="*40)

class HumanAgent:
    def __init__(self, num_actions):
        self.use_raw = False
        
    def step(self, state):
        print_state(state)
        action_ids = list(state['legal_actions'].keys())
        raw_actions = state['raw_legal_actions']
        
        print("\nВаши доступные ходы:")
        for i, action in enumerate(raw_actions):
            pretty_action = action.replace('r', 'Красн').replace('b', 'Син').replace('g', 'Зел').replace('y', 'Желт')
            print(f"  [{i}] -> {pretty_action}")
            
        while True:
            try:
                choice = int(input(f"Ваш выбор (введите цифру 0-{len(raw_actions)-1}): "))
                if 0 <= choice < len(raw_actions):
                    print("-" * 40)
                    return action_ids[choice]
                else:
                    print("❌ Нет такого варианта!")
            except ValueError:
                print("❌ Пожалуйста, введите цифру.")

    def eval_step(self, state):
        return self.step(state), {}

def main():
    if not os.path.exists('models/uno_ai_model.pth'):
        print("❌ Файл с мозгом ИИ не найден! Сначала обучите его командой: python train.py")
        return

    print("🤖 Загрузка ИИ...")
    # Создаем игру
    env = rlcard.make('uno', config={'game_num_players': 2})
    
    # Инициализируем ИИ
    device = get_device()
    agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[64, 64],
        device=device
    )
    
    # Загружаем сохраненный "мозг"
    agent.q_estimator.qnet.load_state_dict(torch.load('models/uno_ai_model.pth', map_location=device))
    agent.use_raw = False

    human_agent = HumanAgent(env.num_actions)
    
    # Игрок 0 - ИИ, Игрок 1 - Человек
    env.set_agents([agent, human_agent])

    print("\n🎮 ИГРА НАЧИНАЕТСЯ! Вы (Игрок 1) против ИИ (Игрок 0).")
    print("Цель игры: скинуть все свои карты первым.")
    
    # Запускаем партию
    trajectories, payoffs = env.run(is_training=False)
    
    print("\n" + "*"*40)
    print("🏁 ИГРА ОКОНЧЕНА 🏁")
    if payoffs[1] > 0:
        print("🏆 ПОБЕДА! ВЫ ОБЫГРАЛИ НЕЙРОСЕТЬ!")
    elif payoffs[1] < 0:
        print("💀 ПОРАЖЕНИЕ! ИИ ОКАЗАЛСЯ УМНЕЕ!")
    else:
        print("🤝 НИЧЬЯ!")
    print("*"*40)

if __name__ == '__main__':
    main()
