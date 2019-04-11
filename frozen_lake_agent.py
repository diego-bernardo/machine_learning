import gym
from frozen_lake import FrozenLakeEnv
import numpy as np
import copy

env = gym.make('FrozenLake-v0')
#env = FrozenLakeEnv(is_slippery=True)

#=====================================================================
# Abaixo os métodos de avaliação das politicas e aprendizado do agente

# Retorna um array de probabilidade com o valor dos estados
# Tamanha do array = número de estados
def policy_evaluation(env, policy, gamma=1, theta=1e-8):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            Vs = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    Vs += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, np.abs(V[s]-Vs))
            V[s] = Vs
        if delta < theta:
            break
    return V


# evaluate the policy 
random_policy = np.ones([env.nS, env.nA]) / env.nA
V = policy_evaluation(env, random_policy)


# Retorna um array de probabilidade
# Qual acao é a melhor dado o estado
# Tamanha do array = número de acoes 
def q_from_v(env, V, s, gamma=1):
    q = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[s][a]:
            q[a] += prob * (reward + gamma * V[next_state])
    return q

Q = np.zeros([env.nS, env.nA])
for s in range(env.nS):
    Q[s] = q_from_v(env, V, s)


# Retorna um array de probabilidades
# do agente escolher determinada ação dado o estado
# Array 2D: [estados][acoes]
def policy_improvement(env, V, gamma=1):
    policy = np.zeros([env.nS, env.nA]) / env.nA
    for s in range(env.nS):
        q = q_from_v(env, V, s, gamma)
        
        # OPTION 1: construct a deterministic policy 
        # policy[s][np.argmax(q)] = 1
        
        # OPTION 2: construct a stochastic policy that puts equal probability on maximizing actions
        best_a = np.argwhere(q==np.max(q)).flatten()
        policy[s] = np.sum([np.eye(env.nA)[i] for i in best_a], axis=0)/len(best_a)
        
    return policy



# Retorna a politica: 
# Array 2D com a probabilidade do agente tomar uma açao dado o estado [estdo][acao]
# Retorna o valor dos estados
# Array 1D com o valor de cada estado
def policy_iteration(env, gamma=1, theta=1e-8):
    policy = np.ones([env.nS, env.nA]) / env.nA
    while True:
        V = policy_evaluation(env, policy, gamma, theta)
        new_policy = policy_improvement(env, V)
        
        # OPTION 1: stop if the policy is unchanged after an improvement step
        if (new_policy == policy).all():
            break;
        
        # OPTION 2: stop if the value function estimates for successive policies has converged
        # if np.max(abs(policy_evaluation(env, policy) - policy_evaluation(env, new_policy))) < theta*1e2:
        #    break;
        
        policy = copy.copy(new_policy)
    return policy, V



# obtain the optimal policy and optimal state-value function
policy_pi, V_pi = policy_iteration(env)

# print the optimal policy
#env.render()
#print("\nOptimal Policy (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3):")
#print(policy_pi,"\n")
#print(V_pi)


#========== MÉTODOS TRUNCADOS ==================
# == FAZEM A MESMA COISA QUE OS MÉTODOS ACIMA ==
# === PORÉM COM UM NUMERO MÁXIMO DE ITERACOES ==

# Retorna um array de probabilidade com o valor dos estados
# Tamanha do array = número de estados
# A diferença para o outro método é que este tem um número máximo de iteracoes
def truncated_policy_evaluation(env, policy, V, max_it=1, gamma=1):
    num_it=0
    while num_it < max_it:
        for s in range(env.nS):
            v = 0
            q = q_from_v(env, V, s, gamma)
            for a, action_prob in enumerate(policy[s]):
                v += action_prob * q[a]
            V[s] = v
        num_it += 1
    return V


# Retorna a politica: 
# Array 2D com a probabilidade do agente tomar uma açao dado o estado [estdo][acao]
# Retorna o valor dos estados
# Array 1D com o valor de cada estado
def truncated_policy_iteration(env, max_it=1, gamma=1, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA]) / env.nA
    while True:
        policy = policy_improvement(env, V)
        old_V = copy.copy(V)
        V = truncated_policy_evaluation(env, policy, V, max_it, gamma)
        if max(abs(V-old_V)) < theta:
            break;
    return policy, V


policy_tpi, V_tpi = truncated_policy_iteration(env, max_it=1000)

# print the optimal policy
#print("\nOptimal Policy (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3):")
#print(policy_tpi,"\n")



def value_iteration(env, gamma=1, theta=1e-8):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            v = V[s]
            V[s] = max(q_from_v(env, V, s, gamma))
            delta = max(delta,abs(V[s]-v))
        if delta < theta:
            break
    policy = policy_improvement(env, V, gamma)
    return policy, V


#====================
# EXECUTANDO O AGENTE

#state = env.reset()
#env.render()
#action = np.argmax(policy_tpi[state])
#state, reward, done, info = env.step(action)
#print(action)
#print(state)
#print(policy_tpi)
#print()
#print(policy_pi)


for i_episode in range(1):
    state = env.reset()
    t = 0
    while True:
        action = np.argmax(policy_pi[state])
        state, reward, done, info = env.step(action)
        env.render()
        print(action)
        t = t+1
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
