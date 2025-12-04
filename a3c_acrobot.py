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
import queue

# ----------------------------------------
# 하이퍼파라미터 및 설정 
# ----------------------------------------
INPUT_SHAPE = 6       # Acrobot 관측 공간 크기 (cos1, sin1, cos2, sin2, vel1, vel2)
N_ACTIONS = 3         # Acrobot 행동 공간 크기 (-1, 0, +1 torque)
MAX_EPISODES = 10000  # 최대 에피소드 수
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.99
LR = 0.0001
EQUILIBRIUM_SCORE = -100  # Acrobot은 -1 보상이 누적되므로, -100 이상이면 성공적인 수준으로 간주
NUM_WORKERS = 16       # 워커 수

# ----------------------------------------
# 경로 설정
# ----------------------------------------
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

# 파일명 변경
EXCEL_FILENAME = os.path.join(BASE_DIR, "acrobot_training_results.xlsx")
GIF_FILENAME = os.path.join(BASE_DIR, "acrobot_demo.gif")

# ----------------------------------------
# 네트워크 정의 (Actor-Critic)
# ----------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(input_dim, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        probs = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return probs, value

# ----------------------------------------
# Shared Optimizer
# ----------------------------------------
class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

# ----------------------------------------
# Worker 클래스
# ----------------------------------------
class Worker(mp.Process):
    def __init__(self, global_net, optimizer, global_ep, global_ep_r, res_queue, name, success_flag, data_queue):
        super(Worker, self).__init__()
        self.name = f"w{name}"
        self.g_ep = global_ep
        self.g_ep_r = global_ep_r
        self.res_queue = res_queue
        self.global_net = global_net
        self.optimizer = optimizer
        self.success_flag = success_flag
        self.data_queue = data_queue

        self.local_net = ActorCritic(INPUT_SHAPE, N_ACTIONS)
        # 환경 변경: Acrobot-v1
        self.env = gym.make('Acrobot-v1')

    def run(self):
        print(f"[{self.name}] Started.")
        state, _ = self.env.reset()
        ep_score = 0
        start_time = time.time()
        
        ep_grad_w2g = 0.0
        ep_grad_g2w = 0.0
        update_count = 0

        while self.g_ep.value < MAX_EPISODES and not self.success_flag.value:
            # 1. 동기화
            self.local_net.load_state_dict(self.global_net.state_dict())

            buffer_s, buffer_a, buffer_r = [], [], []

            # 2. 데이터 수집
            for t in range(UPDATE_GLOBAL_ITER):
                state_tensor = torch.from_numpy(state).float()
                probs, _ = self.local_net(state_tensor)
                
                m = torch.distributions.Categorical(probs)
                action = m.sample().item()

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                ep_score += reward

                
                buffer_s.append(state)
                buffer_a.append(action)
                buffer_r.append(reward)

                state = next_state

                if done:
                    break
            
            # 3. Bootstrap Value 계산
            R = 0
            if not done:
                state_tensor = torch.from_numpy(state).float()
                _, value = self.local_net(state_tensor)
                R = value.item()

            # 4. Loss 계산
            buffer_v_target = []
            for r in buffer_r[::-1]:
                R = r + GAMMA * R
                buffer_v_target.append(R)
            buffer_v_target.reverse()

            s_batch = torch.tensor(np.array(buffer_s), dtype=torch.float)
            a_batch = torch.tensor(np.array(buffer_a), dtype=torch.long).view(-1, 1)
            v_target = torch.tensor(np.array(buffer_v_target), dtype=torch.float).view(-1, 1)

            probs, v = self.local_net(s_batch)
            advantage = v_target - v

            c_loss = advantage.pow(2)
            m = torch.distributions.Categorical(probs)
            a_loss = -m.log_prob(a_batch.squeeze()).unsqueeze(1) * advantage.detach()
            entropy = m.entropy().unsqueeze(1)

            total_loss = (a_loss + c_loss - 0.001 * entropy).mean()

            # 5. 업데이트
            self.optimizer.zero_grad()
            total_loss.backward()
            
            grad_w2g_norm = 0.0
            for p in self.local_net.parameters():
                if p.grad is not None:
                    grad_w2g_norm += p.grad.data.norm(2).item() ** 2
            grad_w2g_norm = grad_w2g_norm ** 0.5
            ep_grad_w2g += grad_w2g_norm

            torch.nn.utils.clip_grad_norm_(self.local_net.parameters(), 0.5)

            for lp, gp in zip(self.local_net.parameters(), self.global_net.parameters()):
                gp._grad = lp.grad
            
            grad_g2w_norm = 0.0
            for p in self.global_net.parameters():
                if p._grad is not None:
                    grad_g2w_norm += p._grad.data.norm(2).item() ** 2
            grad_g2w_norm = grad_g2w_norm ** 0.5
            ep_grad_g2w += grad_g2w_norm
            
            update_count += 1
            self.optimizer.step()

            if done:
                end_time = time.time()
                duration = end_time - start_time
                
                with self.g_ep.get_lock():
                    self.g_ep.value += 1
                    current_ep = self.g_ep.value

                with self.g_ep_r.get_lock():
                    if self.g_ep_r.value == 0:
                        self.g_ep_r.value = ep_score
                    else:
                        self.g_ep_r.value = self.g_ep_r.value * 0.99 + ep_score * 0.01
                    current_avg_score = self.g_ep_r.value

                avg_grad_w2g = ep_grad_w2g / max(1, update_count)
                avg_grad_g2w = ep_grad_g2w / max(1, update_count)

                # 로그 출력
                print(f"{self.name} | Ep: {current_ep} | Score: {int(ep_score)} | Avg: {int(current_avg_score)} | Time: {duration:.2f}s")

                log_data = {
                    "episode": current_ep,
                    "worker": self.name,
                    "grad_w2g": avg_grad_w2g,
                    "grad_g2w": avg_grad_g2w,
                    "score": ep_score,
                    "avg_score": current_avg_score,
                    "time": duration
                }
                self.data_queue.put(log_data)


                if current_avg_score >= EQUILIBRIUM_SCORE:
                    with self.success_flag.get_lock():
                        self.success_flag.value = True
                    print(f"[{self.name}] Target score reached! Signaling termination.")

                ep_score = 0
                ep_grad_w2g = 0.0
                ep_grad_g2w = 0.0
                update_count = 0
                state, _ = self.env.reset()
                start_time = time.time()

        self.env.close()
        print(f"[{self.name}] Finished execution.")

# ----------------------------------------
# 메인 실행 블록
# ----------------------------------------
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    global_net = ActorCritic(INPUT_SHAPE, N_ACTIONS)
    global_net.share_memory()

    opt = SharedAdam(global_net.parameters(), lr=LR)

    global_ep = mp.Value('i', 0)
    global_ep_r = mp.Value('d', 0.)
    success_flag = mp.Value('b', False)
    res_queue = mp.Queue()
    data_queue = mp.Queue()

    print(f"System: Starting training (Acrobot-v1) with {NUM_WORKERS} workers...")
    print(f"System: Saving results to: {BASE_DIR}")
    
    workers = [Worker(global_net, opt, global_ep, global_ep_r, res_queue, i, success_flag, data_queue) for i in range(NUM_WORKERS)]

    for w in workers:
        w.start()

    # 데이터 수집 (데드락 방지 로직 유지)
    data_list = []
    
    while True:
        try:
            while True:
                record = data_queue.get_nowait()
                data_list.append(record)
        except queue.Empty:
            pass

        if all(not w.is_alive() for w in workers):
            try:
                while True:
                    record = data_queue.get_nowait()
                    data_list.append(record)
            except queue.Empty:
                pass
            break
        
        time.sleep(0.1)

    for w in workers:
        w.join()

    print("System: Training finished. Processing data...")

    # 1. 엑셀 저장
    print(f"System: Converting {len(data_list)} log records to DataFrame...")
    
    if data_list:
        try:
            df = pd.DataFrame(data_list)
            df = df.sort_values(by="episode")
            
            print(f"System: Saving Excel file to {EXCEL_FILENAME}...")
            df.to_excel(EXCEL_FILENAME, index=False)
            print("System: Excel save SUCCESS.")
        except Exception as e:
            print(f"System: Excel save FAILED. Error: {e}")
    else:
        print("System: No data collected to save.")

    # 2. 결과 GIF 생성 (Acrobot)
    if success_flag.value:
        print("System: Equilibrium reached! Recording GIF for Acrobot...")
        try:
            # 환경 변경: Acrobot-v1
            env = gym.make('Acrobot-v1', render_mode='rgb_array')
            state, _ = env.reset()
            done = False
            total_reward = 0
            frames = []

            while not done:
                frame = env.render()
                frames.append(frame)
                
                state_tensor = torch.from_numpy(state).float()
                probs, _ = global_net(state_tensor)
                action = torch.argmax(probs).item()
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += reward

            env.close()
            
            print(f"System: Saving GIF to {GIF_FILENAME}...")
            imageio.mimsave(GIF_FILENAME, frames, fps=30)
            print(f"System: GIF saved with score {total_reward}")
        except Exception as e:
            print(f"System: GIF generation failed. Error: {e}")
    else:
        print("System: Failed to reach equilibrium. Skipping GIF generation.")