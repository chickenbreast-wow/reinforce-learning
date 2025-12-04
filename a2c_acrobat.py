import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
import time
import imageio
import os

# ----------------------------------------
# 하이퍼파라미터 (Reward Shaping 적용 버전)
# ----------------------------------------
INPUT_SHAPE = 6
N_ACTIONS = 3
MAX_EPISODES = 1000
N_STEPS = 20          # 짧은 호흡으로 자주 업데이트 (Shaping이 있으므로 즉각 피드백)
GAMMA = 0.99
LR = 0.0007           # 적절한 학습률
ENTROPY_BETA = 0.01   
EQUILIBRIUM_SCORE = -100
NUM_WORKERS = 4

# ----------------------------------------
# 경로 설정
# ----------------------------------------
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

EXCEL_FILENAME = os.path.join(BASE_DIR, "acrobot_shaped_results.xlsx")
GIF_FILENAME = os.path.join(BASE_DIR, "acrobot_shaped_demo.gif")

# ----------------------------------------
# 네트워크 정의
# ----------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(input_dim, 256)
        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        probs = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return probs, value

# ----------------------------------------
# EnvRunner (핵심: Reward Shaping 추가)
# ----------------------------------------
class EnvRunner(mp.Process):
    def __init__(self, pipe, worker_id):
        super(EnvRunner, self).__init__()
        self.pipe = pipe
        self.worker_id = worker_id
        self.env = gym.make('Acrobot-v1')

    def run(self):
        state, _ = self.env.reset()
        ep_score = 0
        
        while True:
            cmd, data = self.pipe.recv()
            
            if cmd == 'step':
                action = data
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
  
                done_for_training = terminated 
                real_done = terminated or truncated
                
                cos1, sin1, cos2, sin2, vel1, vel2 = next_state

                height = -cos1 - cos2 

                shaping_reward = reward + (height * 0.5) 
                
                # 성공 시(terminated) 큰 보너스
                if terminated:
                    shaping_reward += 50.0

                # 스케일링 (학습 안정화)
                scaled_reward = shaping_reward / 10.0

                ep_score += reward # 로그에는 원본 점수(-1 누적) 기록
                
                info = {}
                if real_done:
                    info['ep_score'] = ep_score
                    ep_score = 0
                    next_state, _ = self.env.reset()
                
                self.pipe.send((next_state, scaled_reward, done_for_training, info))
                
            elif cmd == 'reset':
                state, _ = self.env.reset()
                ep_score = 0
                self.pipe.send(state)
                
            elif cmd == 'close':
                self.env.close()
                break

# ----------------------------------------
# 메인 실행 블록
# ----------------------------------------
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ActorCritic(INPUT_SHAPE, N_ACTIONS).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    print(f"System: Starting Shaped A2C Training (Acrobot-v1) with {NUM_WORKERS} workers...")
    print("System: Applied Reward Shaping (Height Bonus) & Correct Truncation Handling.")
    
    parent_conns, child_conns = zip(*[mp.Pipe() for _ in range(NUM_WORKERS)])
    workers = [EnvRunner(child_conn, i) for i, child_conn in enumerate(child_conns)]
    
    for w in workers:
        w.start()

    for parent_conn in parent_conns:
        parent_conn.send(('reset', None))
    states = [parent_conn.recv() for parent_conn in parent_conns]
    states = np.stack(states)

    total_episodes = 0
    avg_score = -500.0
    log_data = []
    start_time = time.time()
    success = False

    while total_episodes < MAX_EPISODES:
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropies = []
        
        for _ in range(N_STEPS):
            state_tensor = torch.FloatTensor(states).to(device)
            probs, value = net(state_tensor)
            
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            
            for parent_conn, action in zip(parent_conns, actions.cpu().numpy()):
                parent_conn.send(('step', action))
            
            next_states, step_rewards, dones, infos = [], [], [], []
            for parent_conn in parent_conns:
                ns, r, d, info = parent_conn.recv()
                next_states.append(ns)
                step_rewards.append(r)
                dones.append(d)
                infos.append(info)
                
                if 'ep_score' in info:
                    total_episodes += 1
                    score = info['ep_score']
                    
                    if avg_score == -500.0:
                        avg_score = score
                    else:
                        avg_score = 0.99 * avg_score + 0.01 * score
                    
                    duration = time.time() - start_time
                    if total_episodes % 20 == 0:
                        print(f"Ep: {total_episodes} | Score: {int(score)} | Avg: {int(avg_score)} | Time: {duration:.2f}s")
                    
                    log_data.append({
                        "episode": total_episodes,
                        "score": score,
                        "avg_score": avg_score,
                        "time": duration
                    })

                    if avg_score >= EQUILIBRIUM_SCORE:
                        success = True

            next_states = np.stack(next_states)
            states = next_states

            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()
            
            log_probs.append(log_prob)
            values.append(value.squeeze(1))
            rewards.append(torch.FloatTensor(step_rewards).to(device))
            masks.append(torch.FloatTensor(1 - np.array(dones)).to(device))
            entropies.append(entropy)
            
            if success:
                break
        
        if success:
            print("System: Target score reached! Stopping training.")
            break

        # Loss Calculation
        next_state_tensor = torch.FloatTensor(next_states).to(device)
        _, next_value = net(next_state_tensor)
        returns = next_value.squeeze(1)
        
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        rewards = torch.stack(rewards)
        masks = torch.stack(masks)
        entropies = torch.stack(entropies)
        
        actor_loss = 0
        critic_loss = 0
        gae = 0
        
        for step in reversed(range(N_STEPS)):
            delta = rewards[step] + GAMMA * returns * masks[step] - values[step]
            gae = delta + GAMMA * 0.95 * masks[step] * gae
            returns = values[step] + gae
            
            advantage = gae.detach()
            
            actor_loss += -(log_probs[step] * advantage)
            critic_loss += F.mse_loss(values[step], returns.detach())
        
        actor_loss = actor_loss.mean()
        critic_loss = critic_loss.mean()
        entropy_loss = entropies.mean()
        
        total_loss = actor_loss + 0.5 * critic_loss - ENTROPY_BETA * entropy_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optimizer.step()

    # 종료 처리
    for parent_conn in parent_conns:
        parent_conn.send(('close', None))
    for w in workers:
        w.join()

    print("System: Training finished.")
    
    if log_data:
        try:
            df = pd.DataFrame(log_data)
            df.to_excel(EXCEL_FILENAME, index=False)
            print("System: Excel saved.")
        except:
            pass

    if success:
        print("System: Recording GIF...")
        try:
            env = gym.make('Acrobot-v1', render_mode='rgb_array')
            state, _ = env.reset()
            done = False
            frames = []
            net.cpu()

            while not done:
                frame = env.render()
                frames.append(frame)
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                probs, _ = net(state_tensor)
                action = torch.argmax(probs).item()
                next_state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = next_state

            env.close()
            imageio.mimsave(GIF_FILENAME, frames, fps=30)
            print("System: GIF saved.")
        except Exception as e:
            print(f"Error: {e}")