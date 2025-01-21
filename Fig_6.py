import numpy as np
import matplotlib.pyplot as plt

class NonlinearSystem:
    def __init__(self):
        self.output = 0.
    def forward(self, input_):
        self.output = self.output/(1+self.output**2) + input_**3
        return self.output
    
class EchoStateNetwork:
    def __init__(self, input_size, reservoir_size, spectral_radius=0.95, alpha=0.9):
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.alpha = alpha
        self.W_in = np.random.rand(reservoir_size, input_size) * 2 - 1
        self.W = np.random.rand(reservoir_size, reservoir_size) - 0.5
        
        rho_W = np.max(np.abs(np.linalg.eigvals(self.W)))
        self.W *= spectral_radius / rho_W
        self.state = np.zeros(reservoir_size)
        
    def forward(self, x):
        u = (self.W_in@x).flatten()
        self.state = ((1 - self.alpha) * self.state + self.alpha * np.tanh(self.W@self.state + u))
        return self.state
    
class RLS:
    def __init__(self, input_size, output_size, delta=1.0, lambda_=0.99):
        self.delta = delta
        self.P = np.eye(input_size) / delta
        self.w_out = np.zeros((output_size, input_size))
        self.lambda_ = lambda_

    def update(self, state, target):
        state = state.reshape(-1, 1)
        target = target.reshape(-1, 1)
        P_state = self.P @ state
        error = self.w_out @ state - target
        k = P_state / (self.lambda_ + state.T @ P_state)
        self.P = (self.P - k @ state.T @ self.P) / self.lambda_
        self.w_out -= np.outer(error, k.T)

    def predict(self, state):
        return self.w_out @ state.reshape(-1, 1)
    

sample = 200
duration = 42

time = np.arange(0, duration, 1/sample)

peak = 20.
bottom = 0.

reference_signal = np.sin(time*0.5*2*np.pi-0.5*np.pi)
reference_signal = (peak-bottom)*(reference_signal-reference_signal.min())/(reference_signal.max()-reference_signal.min())+bottom

delta_ = 5

Kp = 1e-4
Kd = 1e-6

noise = 0.1

reservoir_size = 50
spectral_radius = 0.8
leaky = 0.8

learning_rate = 1e0
forgetting_factor = 1-1e-6

repeat = 100

################################################################

outputs_esn = np.zeros((time.shape[0], repeat))

for seed in range (repeat):
    np.random.seed(1234567*(seed+1))

    nonlinear_system_esn = NonlinearSystem()

    esn = EchoStateNetwork(1, reservoir_size, spectral_radius, leaky)
    rls_esn = RLS(reservoir_size+delta_, 1, learning_rate, forgetting_factor)

    washout = 100
    for _ in range(washout):
        esn.forward(np.array([0]).reshape(-1,1))

    y_esn = np.zeros_like(time)
    y_esn_feedback = np.zeros_like(time)
    u_esn = np.zeros_like(time)
    last_u_ff_esn = 0
    esn_states = []
    for t in range(delta_):
        error = reference_signal[t]-y_esn_feedback[t]
        if t > 0:
            error_dif = (reference_signal[t]-reference_signal[t-1]+y_esn_feedback[t-1]-y_esn_feedback[t])*sample
        else:
            error_dif = 0
            
        esn_state = esn.forward(reference_signal[t+delta_].reshape(-1,1))
        esn_states.append(esn_state)
        
        u_fb = Kp * error + Kd * error_dif
        u_esn[t] = u_esn[t-1] + u_fb
        y_esn[t+1] = nonlinear_system_esn.forward(u_esn[t])
        y_esn_feedback[t+1] = y_esn[t+1] + np.random.normal(0, noise)

    for t in range(delta_, duration*sample-delta_):
        esn_state = esn.forward(reference_signal[t+delta_].reshape(-1,1))
        esn_states.append(esn_state)
        
        extended_learning_state = np.concatenate((esn_states[-delta_], y_esn_feedback[t-delta_+1:t+1]))
            
        rls_esn.update(extended_learning_state, u_esn[t-delta_])
        
        extended_control_state = np.concatenate((esn_state, reference_signal[t+1:t+delta_+1]))

        u_ff = rls_esn.predict(extended_control_state).item()
        
        error = reference_signal[t]-y_esn_feedback[t]
        error_dif = (reference_signal[t]-reference_signal[t-1]+y_esn_feedback[t-1]-y_esn_feedback[t])*sample
        u_fb = Kp * error + Kd * error_dif
        u_esn[t] = u_esn[t-1] + (u_ff-last_u_ff_esn) + u_fb

        last_u_ff_esn = u_ff
        
        y_esn[t+1] = nonlinear_system_esn.forward(u_esn[t])
        y_esn_feedback[t+1] = y_esn[t+1] + np.random.normal(0, noise)

    outputs_esn[:, seed] = y_esn 
    print(f'Running simulation using ESN+PD {seed+1}/100 completed')

mean_outputs_esn = np.mean(outputs_esn, axis=1)
std_outputs_esn = np.std(outputs_esn, axis=1)

################################################################

outputs_linear = np.zeros((time.shape[0], repeat))

for seed in range (repeat):
    np.random.seed(1234567*(seed+1))

    nonlinear_system_linear = NonlinearSystem()

    rls_linear =  RLS(delta_, 1, learning_rate, forgetting_factor)

    y_linear = np.zeros_like(time)
    y_linear_feedback = np.zeros_like(time)
    u_linear = np.zeros_like(time)
    last_u_ff_linear = 0

    for t in range(delta_):
        error = reference_signal[t]-y_linear_feedback[t]
        if t > 0:
            error_dif = (reference_signal[t]-reference_signal[t-1]+y_linear_feedback[t-1]-y_linear_feedback[t])*sample
        else:
            error_dif = 0    

        u_fb = Kp * error + Kd * error_dif
        u_linear[t] = u_linear[t-1] + u_fb
        y_linear[t+1] = nonlinear_system_linear.forward(u_linear[t])
        y_linear_feedback[t+1] = y_linear[t+1] + np.random.normal(0, noise)
        
    for t in range(delta_, duration*sample-delta_):
            
        rls_linear.update(y_linear_feedback[t-delta_+1:t+1], u_linear[t-delta_])
        u_ff = rls_linear.predict(reference_signal[t+1:t+delta_+1]).item()
        
        error = reference_signal[t]-y_linear_feedback[t]
        error_dif = (reference_signal[t]-reference_signal[t-1]+y_linear_feedback[t-1]-y_linear_feedback[t])*sample
        u_fb = Kp * error + Kd * error_dif
        u_linear[t] = u_linear[t-1] + (u_ff-last_u_ff_linear) + u_fb

        last_u_ff_linear = u_ff
        
        y_linear[t+1] = nonlinear_system_linear.forward(u_linear[t])
        y_linear_feedback[t+1] = y_linear[t+1] + np.random.normal(0, noise)

    outputs_linear[:, seed] = y_linear
    print(f'Running simulation using Linear+PD {seed+1}/100 completed')

mean_outputs_linear = np.mean(outputs_linear, axis=1)
std_outputs_linear = np.std(outputs_linear, axis=1)

################################################################

outputs_pd = np.zeros((time.shape[0], repeat))

for seed in range (repeat):
    np.random.seed(1234567*(seed+1))

    nonlinear_system_pd = NonlinearSystem()

    y_pd = np.zeros_like(time)
    y_pd_feedback = np.zeros_like(time)
    u_pd = np.zeros_like(time)

    for t in range(duration*sample-1):
        error = reference_signal[t]-y_pd_feedback[t]
        if t > 0:
            error_dif = (reference_signal[t]-reference_signal[t-1]+y_pd_feedback[t-1]-y_pd_feedback[t])*sample
        else:
            error_dif = 0    

        u_fb = Kp * error + Kd * error_dif
        u_pd[t] = u_pd[t-1] + u_fb
        y_pd[t+1] = nonlinear_system_pd.forward(u_pd[t])
        y_pd_feedback[t+1] = y_pd[t+1] + np.random.normal(0, noise)

    outputs_pd[:, seed] = y_pd
    print(f'Running simulation using PD {seed+1}/100 completed')

mean_outputs_pd = np.mean(outputs_pd, axis=1)
std_outputs_pd = np.std(outputs_pd, axis=1)

################################################################

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 7))

plot_span = 40*sample

ax1.plot(time[:plot_span] * sample / 1000, reference_signal[:plot_span], label='Reference', color='black', linewidth=2)
ax1.plot(time[:plot_span] * sample / 1000, mean_outputs_esn[:plot_span], label='ESN+PD', color='red', linewidth=1)
ax1.fill_between(time[:plot_span]*sample/1000, (mean_outputs_esn-std_outputs_esn)[:plot_span], (mean_outputs_esn+std_outputs_esn)[:plot_span], color='red', alpha=0.2, edgecolor='none')
ax1.set_xlabel('Step/1000 (-)')
ax1.set_ylabel('Output (-)')
ax1.legend()

ax2.plot(time[:plot_span] * sample / 1000, reference_signal[:plot_span], label='Reference', color='black', linewidth=2)
ax2.plot(time[:plot_span] * sample / 1000, mean_outputs_linear[:plot_span], label='Linear+PD', color='green', linewidth=1)
ax2.fill_between(time[:plot_span]*sample/1000, (mean_outputs_linear-std_outputs_linear)[:plot_span], (mean_outputs_linear+std_outputs_linear)[:plot_span], color='green', alpha=0.2, edgecolor='none')
ax2.set_xlabel('Step/1000 (-)')
ax2.set_ylabel('Output (-)')
ax2.legend()

ax3.plot(time[:plot_span] * sample / 1000, reference_signal[:plot_span], label='Reference', color='black', linewidth=2)
ax3.plot(time[:plot_span] * sample / 1000, mean_outputs_pd[:plot_span], label='PD', color='orange', linewidth=1)
ax3.fill_between(time[:plot_span]*sample/1000, (mean_outputs_pd-std_outputs_pd)[:plot_span], (mean_outputs_pd+std_outputs_pd)[:plot_span], color='orange', alpha=0.2, edgecolor='none')

ax3.set_xlabel('Step/1000 (-)')
ax3.set_ylabel('Output (-)')
ax3.legend()

ax4.plot(time[:plot_span] * sample / 1000, (reference_signal - mean_outputs_pd)[:plot_span], label='PD', color='orange', linewidth=1)
ax4.plot(time[:plot_span] * sample / 1000, (reference_signal - mean_outputs_linear)[:plot_span], label='Linear+PD', color='green', linewidth=1)
ax4.plot(time[:plot_span] * sample / 1000, (reference_signal - mean_outputs_esn)[:plot_span], label='ESN+PD', color='red', linewidth=1)

ax4.fill_between(time[:plot_span]*sample/1000, (reference_signal-mean_outputs_esn-std_outputs_esn)[:plot_span], (reference_signal-mean_outputs_esn+std_outputs_esn)[:plot_span], color='red', alpha=0.2, edgecolor='none')
ax4.fill_between(time[:plot_span]*sample/1000, (reference_signal-mean_outputs_linear-std_outputs_linear)[:plot_span], (reference_signal-mean_outputs_linear+std_outputs_linear)[:plot_span], color='green', alpha=0.2, edgecolor='none')
ax4.fill_between(time[:plot_span]*sample/1000, (reference_signal-mean_outputs_pd-std_outputs_pd)[:plot_span], (reference_signal-mean_outputs_pd+std_outputs_pd)[:plot_span], color='orange', alpha=0.2, edgecolor='none')

ax4.set_xlabel('Step/1000 (-)')
ax4.set_ylabel('Error (-)')
ax4.legend()

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlim(0, 6)
    ax.xaxis.set_tick_params(direction='in', top=True)
    ax.yaxis.set_tick_params(direction='in', right=True)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

ax1.set_ylim(-2, 22)
ax2.set_ylim(-2, 22)
ax3.set_ylim(-2, 22)
ax4.set_ylim(-15, 15)

plt.tight_layout()
plt.show()