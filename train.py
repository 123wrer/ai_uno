import rlcard
from rlcard.agents import DQNAgent, RandomAgent
from rlcard.utils import reorganize, tournament
import torch
import matplotlib.pyplot as plt
import os

def get_compute_device():
    # Проверяем поддержку CUDA (для видеокарт Nvidia)
    if torch.cuda.is_available():
        print(f"✅ Найдена видеокарта Nvidia: {torch.cuda.get_device_name(0)}")
        print("🚀 Используем CUDA для максимальной скорости обучения!")
        return torch.device("cuda")
        
    # Проверяем доступность Vulkan (например, для RX 590 или других видеокарт)
    elif hasattr(torch, 'is_vulkan_available') and torch.is_vulkan_available():
        print("✅ Найден Vulkan! Пытаемся использовать его...")
        try:
            _ = torch.tensor([1.0]).to("vulkan")
            return torch.device("vulkan")
        except Exception as e:
            print(f"⚠️ Ошибка инициализации Vulkan для тензоров: {e}")
            print("🔄 Переключаемся на CPU...")
            return torch.device("cpu")
            
    # Если ничего нет, используем процессор
    else:
        print("ℹ️ CUDA (Nvidia) и Vulkan не найдены или не поддерживаются для обучения.")
        print("🚀 Используем процессор (CPU). Для УНО его мощности хватит с запасом!")
        return torch.device("cpu")

def main():
    print("--- Инициализация игры УНО ---")
    # 1. Создаем среду
    env = rlcard.make('uno', config={'game_num_players': 2})

    # 2. Инициализируем ИИ
    device = get_compute_device()
    agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[64, 64],
        device=device
    )

    # 3. Второй игрок - случайный бот (Среда сама управляет им, мы не знаем его карт!)
    random_agent = RandomAgent(num_actions=env.num_actions)
    env.set_agents([agent, random_agent])

    train_episodes = 20000
    evaluate_every = 1000
    win_rates = []
    episodes_list = []

    print("\n--- Начинаем обучение ИИ ---")
    for episode in range(train_episodes):
        # Играем против среды. ИИ играет, не зная логику второго игрока.
        trajectories, payoffs = env.run(is_training=True)
        trajectories = reorganize(trajectories, payoffs)
        
        # ИИ обучается ТОЛЬКО на своих действиях (trajectories[0])
        # Он анализирует: "Я походил так-то, в итоге проиграл/выиграл. Делаю выводы".
        for ts in trajectories[0]:
            agent.feed(ts)
            
        if episode % evaluate_every == 0:
            reward = tournament(env, 100)[0]
            win_rate = (reward + 1) / 2 * 100
            win_rates.append(win_rate)
            episodes_list.append(episode)
            print(f"Сыграно партий: {episode:4} | Процент побед ИИ: {win_rate:.1f}%")

    print("\n✅ Обучение завершено!")
    
    # Сохраняем модель
    os.makedirs('models', exist_ok=True)
    torch.save(agent.q_estimator.qnet.state_dict(), 'models/uno_ai_model.pth')
    print("💾 Мозг ИИ сохранен в папку 'models/uno_ai_model.pth'")

    # Сохраняем график (чтобы не блокировать терминал, сохраняем как картинку)
    plt.figure(figsize=(8, 5))
    plt.plot(episodes_list, win_rates, marker='o', color='b', linewidth=2)
    plt.title('Прогресс ИИ в УНО (CUDA/Vulkan/CPU)')
    plt.xlabel('Количество партий')
    plt.ylabel('Процент побед (%)')
    plt.grid(True)
    plt.savefig('training_progress.png')
    print("📊 График обучения сохранен в файл 'training_progress.png'")

if __name__ == '__main__':
    main()
