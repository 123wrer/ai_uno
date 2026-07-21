import os
from time import sleep
import torch
import rlcard
from rlcard.agents import DQNAgent
from rlcard.utils import get_device

# Импортируем библиотеку для красивого интерфейса в терминале
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

def translate_card(card_str):
    """Переводит внутренние названия карт RLCard на красивый русский язык с цветами"""
    COLORS = {
        'r': ('Красный', 'red'), 
        'b': ('Синий', 'blue'), 
        'g': ('Зеленый', 'green'), 
        'y': ('Желтый', 'yellow')
    }
    TRAITS = {
        'skip': 'Пропуск хода 🚫', 
        'reverse': 'Смена направления 🔄', 
        'draw_2': 'Возьми две +2 🃏'
    }
    
    if card_str == 'draw':
        return Text("📥 Взять карту из колоды", style="bold white")
    elif card_str == 'wild':
        return Text("🌈 Дикая карта (Смена цвета)", style="bold magenta")
    elif card_str == 'wild_draw_4':
        return Text("🌈 Дикая +4", style="bold magenta")
    elif card_str.startswith('wild-'):
        c = card_str.split('-')[1]
        name, color = COLORS.get(c, (c, 'white'))
        return Text(f"🌈 Дикая (Выбрать цвет: {name})", style=f"bold {color}")
    elif card_str.startswith('wild_draw_4-'):
        c = card_str.split('-')[1]
        name, color = COLORS.get(c, (c, 'white'))
        return Text(f"🌈 Дикая +4 (Выбрать цвет: {name})", style=f"bold {color}")
    elif '-' in card_str:
        c, t = card_str.split('-', 1)
        name, color = COLORS.get(c, (c, 'white'))
        trait = TRAITS.get(t, t)
        return Text(f"{name} {trait}", style=f"bold {color}")
    elif card_str in COLORS:
        name, color = COLORS[card_str]
        return Text(f"Заказан цвет: {name}", style=f"bold {color}")
    else:
        return Text(card_str, style="white")

class HumanRichAgent:
    def __init__(self, num_actions):
        self.use_raw = False
        
    def step(self, state):
        # Очищаем экран для красивой отрисовки нового хода
        console.clear()
        
        raw_obs = state['raw_obs']
        action_ids = list(state['legal_actions'].keys())
        raw_actions = state['raw_legal_actions']
        
        # 1. Отрисовка карты на столе
        target_card = translate_card(raw_obs['target'])
        panel_target = Panel(
            target_card, 
            title="[bold white]🃏 Карта на столе[/]", 
            border_style="cyan", 
            expand=False,
            padding=(1, 5)
        )
        
        # 2. Отрисовка вашей руки
        hand_texts = [translate_card(c) for c in raw_obs['hand']]
        hand_display = Text(" | ").join(hand_texts)
        panel_hand = Panel(
            hand_display, 
            title="[bold white]🖐 Ваша рука[/]", 
            border_style="green", 
            expand=False,
            padding=(1, 2)
        )
        
        console.print(panel_target)
        console.print(panel_hand)
        console.print()
        
        # 3. Таблица возможных ходов
        table = Table(title="Ваши доступные ходы", box=box.ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("Номер", justify="center", style="cyan", no_wrap=True)
        table.add_column("Действие", style="white")
        
        for i, action in enumerate(raw_actions):
            table.add_row(str(i), translate_card(action))
            
        console.print(table)
        
        # Ввод пользователя
        while True:
            choice_str = console.input("\n[bold yellow]Введите номер вашего хода:[/] ")
            try:
                choice = int(choice_str)
                if 0 <= choice < len(raw_actions):
                    return action_ids[choice]
                else:
                    console.print("[bold red]❌ Нет такого варианта, попробуйте снова![/]")
            except ValueError:
                console.print("[bold red]❌ Пожалуйста, введите цифру![/]")

    def eval_step(self, state):
        return self.step(state), {}

def main():
    if not os.path.exists('models/uno_ai_model.pth'):
        console.print("[bold red]❌ Модель ИИ не найдена! Сначала запустите train.py[/]")
        return

    console.print("[bold cyan]🤖 Загрузка мозга ИИ...[/]")
    env = rlcard.make('uno', config={'game_num_players': 2})
    device = get_device()
    
    agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[64, 64],
        device=device
    )
    
    agent.q_estimator.qnet.load_state_dict(torch.load('models/uno_ai_model.pth', map_location=device))
    agent.use_raw = False
    
    human_agent = HumanRichAgent(env.num_actions)
    
    # 0 - ИИ, 1 - Вы
    env.set_agents([agent, human_agent])

    console.print(Panel.fit(
        "[bold green]🎮 ИГРА НАЧИНАЕТСЯ! Вы против ИИ.[/]\nЦель игры: скинуть все свои карты первым.", 
        border_style="bold yellow"
    ))
    sleep(2)
    
    # Запуск партии
    trajectories, payoffs = env.run(is_training=False)
    
    # Итоги
    console.clear()
    if payoffs[1] > 0:
        result_msg = "[bold green]🏆 ПОБЕДА! ВЫ ОБЫГРАЛИ НЕЙРОСЕТЬ![/]"
    elif payoffs[1] < 0:
        result_msg = "[bold red]💀 ПОРАЖЕНИЕ! ИИ ОКАЗАЛСЯ УМНЕЕ![/]"
    else:
        result_msg = "[bold yellow]🤝 НИЧЬЯ![/]"
        
    console.print(Panel.fit(
        result_msg, 
        title="[bold white]🏁 ИГРА ОКОНЧЕНА 🏁[/]", 
        border_style="bold magenta",
        padding=(2, 5)
    ))

if __name__ == '__main__':
    main()
