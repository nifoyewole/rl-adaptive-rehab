# Adaptive Reinforcement Learning for Personalised Stroke Rehabilitation
### A Cost-Sensitive Simulation Study

**Module:** Foundations of Artificial Intelligence (H9FAI)  
**Programme:** MSc Artificial Intelligence | MSC AI in Business -- National College of Ireland, 2026  
**Team:** 
1. Axel Adewale Ilenre (24160873)
2. Oluwanifemi Oyeleye Oyewole (25135163)
3. Onyinyechi Miracle Obodoeze (25161474)

---

## Running the experiments
TO RUN THE EXPERIMENT, please see detailed steps in running_steps.txt in project root.

---

## What our project does
We model post-stroke hemiparesis rehabilitation as a Markov Decision Process (MDP).  
A reinforcement learning agent observes a simulated patient's range of motion, movement accuracy, fatigue level, and recovery stage, then decides which therapy intensity to prescribe at each step.  
Intensities: low difficulty, medium difficulty, high difficulty, or rest.

The reward function balances three objectives simultaneously: motor recovery gain, fatigue penalty, and therapy cost.  
Two RL agents (tabular Q-Learning and a Deep Q-Network (DQN)) are trained and compared against three baselines (static schedule, heuristic clinical policy, random policy) across three patient severity profiles: mild, moderate, and severe.  
Our project also includes a reward function ablation study

## Research Questions:  
1. Can RL outperform static and heuristic therapy protocols in cumulative motor recovery?
2. How does incorporating fatigue dynamics affect policy stability and long-term recovery?
3. Does cost-sensitive reward shaping reduce resource utilisation without significant performance loss?
4. Do learned RL policies generalise across mild, moderate, and severe severity profiles?

---

## Markov Decision Process formulation summary
State: [Range of motion score (ROM), accuracy, fatigue level, recovery stage] — all normalised to [0, 1]  
Actions: {0: low difficulty, 1: medium difficulty, 2: high difficulty, 3: rest}  
Reward: R = 2.0 · Δrecovery − 0.3 · fatigue_penalty − 0.1 · cost_penalty, with a +20 terminal bonus on recovery threshold achievement  
Recovery threshold: ROM ≥ 85 and accuracy ≥ 80 within 50 steps  
Discount factor: 0.95  (we did this to make the agent think long-term)

---

## Project structure

```
ca-adaptive-rl-rehab/
|-- src/
|   |-- env/
|   |   |-- patient_profiles.py     patient severity configurations
|   |   |-- patient_env.py          MDP simulation environment (Gymnasium)
|   |-- agents/
|   |   |-- baselines.py            static, heuristic, and random policies
|   |   |-- q_learning.py           tabular Q-learning agent
|   |   |-- dqn.py                  Deep Q-Network agent (PyTorch)
|   |-- training/
|   |   |-- train.py                experiment runner for all agents
|   |   |-- ablation.py             reward weight ablation study
|   |-- evaluation/
|       |-- plots.py                chart generation and business metrics
|-- results/                        generated outputs (JSON, PNG, model files)
|-- requirements.txt
|-- running_steps.txt
|-- README.md

```

---

## Interpreting the results folder
After all three steps, the results/ folder contains the following:

JSON files hold the raw numerical results for each profile and the ablation study, including per-episode reward lists, mean reward, success rate, mean final ROM, mean final fatigue, and business metrics.

PNG charts are sized for presentation slides (1920×1080). The learning curves show the smoothed episode reward over training alongside dashed baseline reference lines. The policy comparison charts show all five policies side by side on mean reward and recovery success rate. The cross-profile chart shows how DQN and Q-Learning perform as patient severity increases.

The key finding to look for: on the moderate profile, DQN achieves approximately 40% recovery success versus the static baseline's 22%, roughly doubling the recovery rate. On the severe profile, Q-Learning collapses to under 1% success while DQN maintains competitive performance — this reflects the limitation of tabular state discretisation in high-complexity state spaces.

In the ablation results, the full model (β=0.3, γ=0.1) intentionally sacrifices some peak success rate compared to the unconstrained recovery-only configuration in exchange for lower and safer fatigue levels.

---
## Research Question Answers  
**RQ1: Can RL outperform static and heuristic protocols?**  
Yes. DQN achieves a 40.8% recovery success rate on the moderate profile versus 22.3% for the  
static baseline - an 85% relative improvement. Q-Learning matches on success rate (41.5%).  
The heuristic policy performs worst overall (19.4%), confirming that conservative rule-based  
fatigue avoidance systematically undershoots therapy intensity. On the mild profile all policies  
converge near 100% (ceiling effect). On severe, DQN (3.7%) outperforms both Static (3.0%) and  
Q-Learning (0.7%).

**RQ2: How does fatigue dynamics affect policy stability?**  
Fatigue penalisation (β=0.3) increases policy variance by 12.3σ units but prevents dangerous  
overtraining. Ablation shows that removing β raises success rate from 39.9% to 52.1% while  
final patient fatigue drops from 98.7 to 81.6, indicating more aggressive, unsafe scheduling.  
The full model deliberately sacrifices peak success rate to maintain patient safety.  

**RQ3: Does cost-sensitive reward shaping reduce resource use without significant loss?**  
Yes. Increasing the cost penalty from γ=0.1 to γ=0.3 reduces success rate from 39.9% to 38.6% (only 1.3) while discouraging high-intensity overuse. Removing cost entirely (γ=0) yields  
only a 0.7pp gain with no efficiency benefit. The current γ=0.1 is the optimal cost-efficiency  
tradeoff.  

**RQ4: Do learned policies generalise across severity profiles?**  
DQN generalises robustly. Mean reward stays stable across mild (57.2), moderate (58.7), and  
severe (30.2) profiles. Q-Learning degrades sharply, collapsing to -6.4 mean reward and 0.7%  
success on the severe profile — worse than random (1.1%). This collapse is caused by 10-bin  
state discretisation losing resolution in the severe profile's low ROM range, where DQN's  
neural function approximation maintains meaningful distinctions.  

## Dependencies  
gymnasium, numpy, torch, matplotlib, scipy, fastapi, uvicorn, websockets  
Please see the requirements.txt  
Our current submission has been modified, and some of these dependencies may not be in use for this version of our work, but you may install them all as you test the project.
