import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import imageio
import os
import time  # 시간 측정을 위해 추가
from collections import deque

# ----------------------------------------
# 하이퍼파라미터 및 설정
# ----------------------------------------
# 환경 설정
INPUT_SHAPE = 4
N_ACTIONS = 2
MAX_EPISODES = 5000
EQUILIBRIUM_SCORE = 400

# DQN 하이퍼파라미터
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE_FREQ = 10  # 에피소드 단위 타겟 업데이트 주기
LR = 0.0005
MEMORY_SIZE = 10000

# 경로 설정 (로컬 실행 기준)
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

EXCEL_FILENAME = os.path.join(BASE_DIR, "training_results_dqn.xlsx")
GIF_FILENAME = os.path.join(BASE_DIR, "cartpole_demo_dqn.gif")

# 장치 설정 (CPU 강제)
device = torch.device("cpu")

# ----------------------------------------
# 네트워크 정의 (Q-Network)
# ----------------------------------------
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.head = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)

# ----------------------------------------
# Replay Buffer (경험 재현 메모리)
# ----------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

# ----------------------------------------
# DQN 에이전트
# ----------------------------------------
class DQNAgent:
    def __init__(self):
        self.policy_net = QNetwork(INPUT_SHAPE, N_ACTIONS).to(device)
        self.target_net = QNetwork(INPUT_SHAPE, N_ACTIONS).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 타겟 넷은 학습하지 않음

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        
        self.epsilon = EPS_START
        self.steps_done = 0

    def select_action(self, state):
        # Epsilon-Greedy Action Selection
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                return self.policy_net(state_t).argmax(dim=1).item()
        else:
            return random.randrange(N_ACTIONS)

    def update_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return None

        # 배포 샘플링
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        state_batch = torch.FloatTensor(states).to(device)
        action_batch = torch.LongTensor(actions).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_state_batch = torch.FloatTensor(next_states).to(device)
        done_batch = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # 현재 상태의 Q값 계산 (Q(s, a))
        curr_q = self.policy_net(state_batch).gather(1, action_batch)

        # 다음 상태의 타겟 Q값 계산 (max Q(s', a'))
        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            target_q = reward_batch + (GAMMA * next_q * (1 - done_batch))

        # Loss 계산 (MSE)
        loss = F.mse_loss(curr_q, target_q)

        # 최적화
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ----------------------------------------
# 메인 학습 루프
# ----------------------------------------
if __name__ == "__main__":
    # 전체 런타임 측정 시작
    total_start_time = time.time()
    
    env = gym.make('CartPole-v1')
    agent = DQNAgent()
    
    # 결과 저장용 리스트
    log_data = []
    
    print(f"System: Starting DQN Training (Local CPU)...")
    print(f"System: Saving results to {BASE_DIR}")

    success = False
    moving_avg_score = 0
    scores_window = deque(maxlen=100) # 최근 100개 에피소드 점수 저장용

    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        score = 0
        total_loss = 0
        train_steps = 0
        
        start_time = time.time()
        
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Reward Shaping
            real_reward = reward
            if done:
                real_reward = -1
            else:
                real_reward = 0.1

            agent.memory.push(state, action, real_reward, next_state, done)
            
            # 학습 수행
            loss = agent.train_step()
            if loss is not None:
                total_loss += loss
                train_steps += 1

            state = next_state
            score += reward
            
            if done:
                break
        
        duration = time.time() - start_time
        
        # 에피소드 종료 후 처리
        agent.update_epsilon()
        scores_window.append(score)
        moving_avg_score = np.mean(scores_window)
        
        # 타겟 네트워크 업데이트
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        # 평균 Loss 계산
        avg_loss = total_loss / train_steps if train_steps > 0 else 0

        # 로그 출력 (Time 추가)
        print(f"Ep: {episode} | Score: {int(score)} | Avg: {moving_avg_score:.1f} | Eps: {agent.epsilon:.2f} | Loss: {avg_loss:.4f} | Time: {duration:.2f}s")

        # 엑셀 저장을 위한 데이터 수집 (Time 추가)
        log_data.append({
            "episode": episode,
            "score": score,
            "avg_score": moving_avg_score,
            "epsilon": agent.epsilon,
            "loss": avg_loss,
            "time": duration
        })

        # 종료 조건 확인
        if moving_avg_score >= EQUILIBRIUM_SCORE:
            print(f"System: Equilibrium reached at episode {episode}!")
            success = True
            break

    env.close()
    
    # 전체 런타임 계산 및 출력
    total_runtime = time.time() - total_start_time
    hours = int(total_runtime // 3600)
    minutes = int((total_runtime % 3600) // 60)
    seconds = total_runtime % 60
    
    print(f"\n{'='*60}")
    print(f"Total Runtime: {hours:02d}:{minutes:02d}:{seconds:05.2f} ({total_runtime:.2f} seconds)")
    print(f"Total Episodes: {episode}")
    print(f"{'='*60}\n")

    # ----------------------------------------
    # 1. 엑셀 저장
    # ----------------------------------------
    if log_data:
        df = pd.DataFrame(log_data)
        df.to_excel(EXCEL_FILENAME, index=False)
        print(f"System: Log saved to {EXCEL_FILENAME}")

    # ----------------------------------------
    # 2. 결과 GIF 생성
    # ----------------------------------------
    if success:
        print("System: Recording GIF...")
        try:
            # 렌더링용 환경 생성
            env = gym.make('CartPole-v1', render_mode='rgb_array')
            state, _ = env.reset()
            done = False
            total_reward = 0
            frames = []

            while not done:
                frame = env.render()
                frames.append(frame)
                
                # 학습된 정책으로 행동 선택 (Epsilon=0, 즉 Greedy)
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = agent.policy_net(state_t)
                    action = q_values.argmax(dim=1).item()
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += reward

            env.close()
            imageio.mimsave(GIF_FILENAME, frames, fps=30)
            print(f"System: GIF saved to {GIF_FILENAME} with score {total_reward}")
        except Exception as e:
            print(f"System: GIF generation failed. Error: {e}")