import numpy as np
import matplotlib.pyplot as plt
import random

#=================== Question 1 ===================#

#========== Input Variables ==========#
num_trials = 201 # number of trials 
t = 25 # in seconds
dt = 0.5 # time step in seconds
time_steps = int(t/dt) # Number of time steps per trial 
stimulus_time = 10 # time of stimulus presentation in seconds
gamma = 1 # currently no discounting
epsilon = 0.2 # learning rate
T_mem = 12 # memory span, in seconds 
mu = 20 # mean of gaussian function
sigma = 1 # standard deviation of gaussian function
memory_steps = int(T_mem / dt) # Discretised memory span 

print("hello world")
#========== Functions ==========#
def g(x): # delta function
    if x == 0:
        return 1
    else:
        return 0

def y(t): # stimulus function
    return g(t-stimulus_time)

def gaussian(x): # normal distribution
    return np.exp(-(x-mu)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

def r(t): # reward function
    return (1/2)*gaussian(t)

#========== Question a ==========#
time_array = np.arange(time_steps) * dt
time = np.arange(0, t+dt, dt)

stimulus_values = list(map(y, time))
reward_values = list(map(r, time))

# UNCOMMENT TO PLOT STIMULUS AND REWARD FUNCTIONS
# plt.plot(time, stimulus_values, label='Stimulus')
# plt.plot(time, reward_values, label='Reward')
# plt.xlabel('Time (s)')  
# plt.ylabel('Stimulus/Reward')
# plt.title('Stimulus and Reward Functions')
# plt.legend()
# plt.show()

#========== Question b ==========#

# 1) Define a helper to get the "feature" vector phi(n), shape = (memory_steps,).
#    phi(n)[tau] = y( t_n - tau*dt )  for tau in 0..(memory_steps-1).
def feature_vector(n):
    # time corresponding to index n is n*dt
    # so phi(n)[tau] = y( (n - tau) * dt )
    # We'll store in a numpy array of length memory_steps.
    phi = np.zeros(memory_steps)
    for tau in range(memory_steps):
        # t_(n - tau) = (n - tau)*dt
        t_val = (n - tau) * dt
        # If (n - tau) < 0, that means "time before 0" => y(t) = 0 anyway
        if t_val < 0:
            phi[tau] = 0.0
        else:
            phi[tau] = y(t_val)
    return phi

# 2) Prepare storage for w, the weight vector for the tapped delay line
w = np.zeros(memory_steps)  # Start with all weights = 0

# We will store certain trials' time courses for plotting. 
# The question states: "For every 10th trial (starting from the first one), plot the key variables."
# That means trials 0,10,20,...,200 (21 total).
plot_trials = range(0, num_trials, 10)

# We will store:
#   V(t)          = value estimate at each time step
#   dV(t)         = gamma*V(t) - V(t-dt)
#   delta(t)      = r(t-dt) + dV(t)
# For each "plot trial", we keep arrays of length time_steps.
all_V = {}
all_dV = {}
all_delta = {}

# Main training loop
for trial in range(num_trials):
    
    # We'll track V(n) for each time step inside this trial
    V = np.zeros(time_steps)
    
    # 1) For n=0, let's define V(0) using w dot feature_vector(0):
    phi_0 = feature_vector(0)
    V[0] = np.dot(w, phi_0)
    
    # 2) For the rest of the time steps:
    for n in range(1, time_steps):
        # Compute the new state's value:
        phi_n   = feature_vector(n)
        V[n]    = np.dot(w, phi_n)
        
        # TD error according to the formula:
        # delta(n) = r(n-1) + gamma*V(n) - V(n-1),
        # but "n-1" means time index (n-1). The reward is at t_{n-1} = (n-1)*dt
        td_error = r((n-1)*dt) + gamma * V[n] - V[n-1]
        
        # Weight update: use the *previous* state's feature vector:
        phi_prev = feature_vector(n-1)
        w += epsilon * td_error * phi_prev
    
    # If this is one of the "plot trials", we compute and store the relevant arrays
    if trial in plot_trials:
        # We want to plot:
        #  (i)   V(t)
        #  (ii)  dV(t) = gamma V(t) - V(t-dt)
        #  (iii) delta(t) = r(t-dt) + dV(t)
        dV = np.zeros_like(V)
        delta_arr = np.zeros_like(V)
        
        dV[0]       = gamma*V[0]  # since V(-1) = 0 for n=0 (no prior time step)
        delta_arr[0] = r(-dt) + dV[0]  # r(-dt)=0, so effectively delta_arr[0] = dV[0]
        
        for n in range(1, time_steps):
            dV[n]        = gamma*V[n] - V[n-1]
            delta_arr[n] = r((n-1)*dt) + dV[n]
        
        all_V[trial]     = V
        all_dV[trial]    = dV
        all_delta[trial] = delta_arr

# Save the TDL dictionaries before they get overwritten by the boxcar code.
td_all_V     = all_V
td_all_dV    = all_dV
td_all_delta = all_delta

# # ========== Plot (b)(i)-(iii): For every 10th trial, plot V(t), dV(t), delta(t) ========== #
# fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# colors = plt.cm.rainbow(np.linspace(0, 1, len(plot_trials)))

# # We'll collect line handles and labels in lists, to build one legend afterward
# all_handles = []
# all_labels = []

# for i, trial_id in enumerate(plot_trials):
#     color = colors[i]
    
#     # 1) Plot V(t) on top axis
#     line_V, = axes[0].plot(time_array, all_V[trial_id],
#                            color=color, label=f'Trial {trial_id}')
    
#     # 2) Plot dV(t) on middle axis
#     axes[1].plot(time_array, all_dV[trial_id],
#                  color=color)  # no new label here
    
#     # 3) Plot delta(t) on bottom axis
#     axes[2].plot(time_array, all_delta[trial_id],
#                  color=color)
    
#     # We only need ONE handle+label per trial to appear in the final legend.
#     all_handles.append(line_V)
#     all_labels.append(f'Trial {trial_id}')

# # Set axes labels/titles
# axes[0].set_ylabel(r'$\hat{V}(t)$')
# axes[0].set_title('Value Estimate over Time')

# axes[1].set_ylabel(r'$\Delta V(t)$ = $\gamma V(t) - V(t-\Delta t)$')
# axes[1].set_title('Temporal Difference of Value')

# axes[2].set_ylabel(r'$\delta(t)$ = $r(t-\Delta t) + \Delta V(t)$')
# axes[2].set_xlabel('Time (s)')
# axes[2].set_title('TD Error')

# # Create a single legend on the right side of the figure
# fig.legend(all_handles, all_labels,    # the lines and labels we collected
#            loc='center left',         # place the legend center-left
#            bbox_to_anchor=(0.8, 0.5))   # at x=1 (right edge), y=0.5 (vertically centered)

# # Make room for the legend so it doesn’t overlap or get cut off
# plt.subplots_adjust(right=0.8)

# plt.show()


##############################################################################
# QUESTION 4 (BOXCAR REPRESENTATION)
##############################################################################
# Use the same time array, same environment, but new representation & new epsilon
# epsilon = 0.01  # smaller learning rate for boxcar representation

def boxcar_feature_vector(n):
    """
    For each tau in [0..memory_steps-1],
    phi[tau] = sum_{u=0..tau} y( (n-u)*dt ), if (n-u) >= 0
    """
    phi = np.zeros(memory_steps)
    for tau in range(memory_steps):
        s = 0.0
        for u in range(tau + 1):
            idx = n - u
            if idx >= 0:
                s += y(idx * dt)
        phi[tau] = s
    return phi

w = np.zeros(memory_steps)  # reset weights

# Trials to plot
plot_trials = range(0, num_trials, 10)
all_V     = {}
all_dV    = {}
all_delta = {}

# MAIN LOOP FOR BOXCAR
for trial in range(num_trials):
    V = np.zeros(time_steps)
    
    # V(0)
    phi_0 = boxcar_feature_vector(0)
    V[0]  = np.dot(w, phi_0)
    
    for n in range(1, time_steps):
        phi_n  = boxcar_feature_vector(n)
        V[n]   = np.dot(w, phi_n)
        
        td_error = r((n-1)*dt) + gamma*V[n] - V[n-1]
        
        phi_prev = boxcar_feature_vector(n-1)
        w += epsilon * td_error * phi_prev
    
    if trial in plot_trials:
        dV_arr     = np.zeros_like(V)
        delta_arr  = np.zeros_like(V)
        dV_arr[0]     = gamma*V[0]
        delta_arr[0]  = r(-dt) + dV_arr[0]
        
        for n in range(1, time_steps):
            dV_arr[n]     = gamma*V[n] - V[n-1]
            delta_arr[n]  = r((n-1)*dt) + dV_arr[n]
        
        all_V[trial]     = V
        all_dV[trial]    = dV_arr
        all_delta[trial] = delta_arr

# ---------- PLOTTING FOR BOXCAR ----------
# fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
# colors = plt.cm.rainbow(np.linspace(0, 1, len(plot_trials)))

# all_handles = []
# all_labels  = []

# for i, trial_id in enumerate(plot_trials):
#     color = colors[i]
    
#     # V(t)
#     line_V, = axes[0].plot(time_array, all_V[trial_id],
#                            color=color, label=f'Trial {trial_id}')
    
#     # dV(t)
#     axes[1].plot(time_array, all_dV[trial_id], color=color)
    
#     # delta(t)
#     axes[2].plot(time_array, all_delta[trial_id], color=color)
    
#     all_handles.append(line_V)
#     all_labels.append(f'Trial {trial_id}')

# axes[0].set_ylabel(r'$\hat{V}(t)$')
# axes[0].set_title('Value Estimate (Boxcar)')

# axes[1].set_ylabel(r'$\Delta V(t)$')
# axes[1].set_title('Temporal Difference of Value (Boxcar)')

# axes[2].set_ylabel(r'$\delta(t)$')
# axes[2].set_xlabel('Time (s)')
# axes[2].set_title('TD Error (Boxcar)')

# fig.legend(all_handles, all_labels, loc='center left', bbox_to_anchor=(0.8, 0.5))
# plt.subplots_adjust(right=0.8)
# plt.show()



# Now save the boxcar dictionaries
boxcar_all_V     = all_V
boxcar_all_dV    = all_dV
boxcar_all_delta = all_delta


##############################################################################
# COMPARISON PLOT: FIRST vs. LAST TRIAL, FOR BOTH REPRESENTATIONS
##############################################################################
# We'll assume the last trial is trial = 200 (since you have 201 trials, indexed 0..200).
# Also assume we stored data for trials in steps of 10 (0,10,20,...,200).

##############################################################################
# COMPARISON PLOT: TDL vs Boxcar at TRIAL 0 vs TRIAL 200
##############################################################################

start_trial = 0
end_trial   = 200

# --- Extract the relevant arrays for Tapped Delay ---
td_V_start    = td_all_V[start_trial]
td_V_end      = td_all_V[end_trial]
td_dV_start   = td_all_dV[start_trial]
td_dV_end     = td_all_dV[end_trial]
td_del_start  = td_all_delta[start_trial]
td_del_end    = td_all_delta[end_trial]

# --- Extract the relevant arrays for Boxcar ---
boxcar_V_start    = boxcar_all_V[start_trial]
boxcar_V_end      = boxcar_all_V[end_trial]
boxcar_dV_start   = boxcar_all_dV[start_trial]
boxcar_dV_end     = boxcar_all_dV[end_trial]
boxcar_del_start  = boxcar_all_delta[start_trial]
boxcar_del_end    = boxcar_all_delta[end_trial]

# time_array is the same time index used in training
# e.g., time_array = np.arange(time_steps) * dt

fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# ===================== 1) V(t) =====================
axes[0].plot(time_array, td_V_start,    label='TDL (Trial 0)')
axes[0].plot(time_array, td_V_end,      label='TDL (Trial 200)')
axes[0].plot(time_array, boxcar_V_start, label='Boxcar (Trial 0)')
axes[0].plot(time_array, boxcar_V_end,   label='Boxcar (Trial 200)')

axes[0].set_ylabel(r'$\hat{V}(t)$')
axes[0].set_title('Value Estimate Comparison')

# ===================== 2) dV(t) =====================
axes[1].plot(time_array, td_dV_start,    label='TDL (Trial 0)')
axes[1].plot(time_array, td_dV_end,      label='TDL (Trial 200)')
axes[1].plot(time_array, boxcar_dV_start, label='Boxcar (Trial 0)')
axes[1].plot(time_array, boxcar_dV_end,   label='Boxcar (Trial 200)')

axes[1].set_ylabel(r'$\Delta V(t)$')
axes[1].set_title('Temporal Difference of Value')

# ===================== 3) delta(t) =====================
axes[2].plot(time_array, td_del_start,    label='TDL (Trial 0)')
axes[2].plot(time_array, td_del_end,      label='TDL (Trial 200)')
axes[2].plot(time_array, boxcar_del_start, label='Boxcar (Trial 0)')
axes[2].plot(time_array, boxcar_del_end,   label='Boxcar (Trial 200)')

axes[2].set_ylabel(r'$\delta(t)$')
axes[2].set_xlabel('Time (s)')
axes[2].set_title('TD Error')

# Create a single legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.9, 0.5))

plt.tight_layout()
plt.suptitle("TDL vs. Boxcar: Trial 0 vs. Trial 200", y=1.03)
plt.show()


##############################################################################
# QUESTION 5 (Partial Reinforcment)
##############################################################################

# Probability of reward on each trial
p_reward      = 0.5        # partial reinforcement


# ================== SATURATING DOPAMINE FUNCTION ================== #
def DA(x, alpha=6.0, x_star=0.27, beta=6.0):
    x = np.asarray(x)  # ensures x can be handled element-wise
    # We can use np.piecewise for vectorized logic:
    return np.piecewise(
        x,
        [x < 0, (x >= 0) & (x < x_star), x >= x_star],
        [lambda z: z/alpha,
         lambda z: z,
         lambda z: x_star + (z - x_star)/beta]
    )

# # ================== INITIALIZE WEIGHTS ================== #
w = np.zeros(memory_steps)

# We store the entire time courses for analysis:
V_matrix     = np.zeros((num_trials, time_steps))
dV_matrix    = np.zeros((num_trials, time_steps))
delta_matrix = np.zeros((num_trials, time_steps))

# Also store a boolean to track whether each trial was rewarded:
is_rewarded  = np.zeros(num_trials, dtype=bool)

# ================== MAIN LEARNING LOOP ================== #
for trial in range(num_trials):
    # Decide if this trial is rewarded (p=0.5)
    rewarded = (np.random.rand() < p_reward)
    is_rewarded[trial] = rewarded
    
    # We'll collect V(t) for this trial
    V = np.zeros(time_steps)
    
    # For n=0
    phi_0 = boxcar_feature_vector(0)
    V[0]  = np.dot(w, phi_0)
    
    # Step through time
    for n in range(1, time_steps):
        phi_n = boxcar_feature_vector(n)
        V[n]  = np.dot(w, phi_n)
        
        # Reward depends on whether this trial is rewarded
        reward_val = r((n-1)*dt) if rewarded else 0.0
        
        # TD error
        td_error = reward_val + gamma*V[n] - V[n-1]
        
        # Update weights based on the previous state's features
        phi_prev = boxcar_feature_vector(n-1)
        w += epsilon * td_error * phi_prev
    
    # Now compute dV(n) and delta(n) for logging
    # dV(n)   = gamma*V(n) - V(n-1)
    # delta(n)= r(t_{n-1}) + dV(n)
    dV_arr     = np.zeros_like(V)
    delta_arr  = np.zeros_like(V)
    
    dV_arr[0]     = gamma*V[0]  # minus V(-1)=0
    delta_arr[0]  = 0.0 + dV_arr[0]  # r(-dt)=0 => delta(0)=dV(0)
    
    for n in range(1, time_steps):
        dV_arr[n]     = gamma*V[n] - V[n-1]
        r_prev        = r((n-1)*dt) if rewarded else 0.0
        delta_arr[n]  = r_prev + dV_arr[n]
    
    V_matrix[trial]     = V
    dV_matrix[trial]    = dV_arr
    delta_matrix[trial] = delta_arr

# ================== (a) PLOT AVERAGES IN LAST 100 TRIALS ================== #
last_100_trials = np.arange(num_trials-100, num_trials)

# Separate rewarded/unrewarded in the last 100
rewarded_mask   = last_100_trials[ is_rewarded[last_100_trials] ]
unrewarded_mask = last_100_trials[~is_rewarded[last_100_trials]]

# Compute mean across time for each group
V_rewarded     = V_matrix[rewarded_mask].mean(axis=0) if len(rewarded_mask)>0 else np.zeros(time_steps)
V_unrewarded   = V_matrix[unrewarded_mask].mean(axis=0) if len(unrewarded_mask)>0 else np.zeros(time_steps)
V_all          = V_matrix[last_100_trials].mean(axis=0)

dV_rewarded    = dV_matrix[rewarded_mask].mean(axis=0) if len(rewarded_mask)>0 else np.zeros(time_steps)
dV_unrewarded  = dV_matrix[unrewarded_mask].mean(axis=0) if len(unrewarded_mask)>0 else np.zeros(time_steps)
dV_all         = dV_matrix[last_100_trials].mean(axis=0)

delta_rewarded   = delta_matrix[rewarded_mask].mean(axis=0) if len(rewarded_mask)>0 else np.zeros(time_steps)
delta_unrewarded = delta_matrix[unrewarded_mask].mean(axis=0) if len(unrewarded_mask)>0 else np.zeros(time_steps)
delta_all        = delta_matrix[last_100_trials].mean(axis=0)

# # --- Plot them ---
# fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# # 1) Value
# axes[0].plot(time_array, V_rewarded,    'g-', label='Rewarded (avg)')
# axes[0].plot(time_array, V_unrewarded,  'r-', label='Unrewarded (avg)')
# axes[0].plot(time_array, V_all,         'b-', label='All (avg)')
# axes[0].set_ylabel(r'$V(t)$')
# axes[0].set_title('Average Value, Last 100 Trials (Boxcar, p=0.5)')

# # 2) dV
# axes[1].plot(time_array, dV_rewarded,   'g-')
# axes[1].plot(time_array, dV_unrewarded, 'r-')
# axes[1].plot(time_array, dV_all,        'b-')
# axes[1].set_ylabel(r'$\Delta V(t)$')

# # 3) delta
# axes[2].plot(time_array, delta_rewarded,   'g-')
# axes[2].plot(time_array, delta_unrewarded, 'r-')
# axes[2].plot(time_array, delta_all,        'b-')
# axes[2].set_ylabel(r'$\delta(t)$')
# axes[2].set_xlabel('Time (s)')

# axes[0].legend(loc='upper right')
# plt.tight_layout()
# plt.show()

# # ================== (c) DOPAMINE SATURATION PLOT ================== #
# # The question asks: "Plot the time course of the dopamine signal averaged
# # across all the last 100 trials. How and why is it different from the average
# # time course of the variable it encodes?"
# # Typically, "the variable it encodes" is the TD error, delta(t).

avg_delta_all = delta_all  # this is the average \delta(t) for all last 100 trials
# But to see how the saturating model acts *trial by trial*, we often do:
#   DA( delta_matrix[trial, :] ) for each trial, then average.
# We'll illustrate the simpler approach: apply DA to each trial's delta, then average:

DA_matrix = np.zeros_like(delta_matrix[last_100_trials])
for i, tr in enumerate(last_100_trials):
    DA_matrix[i] = DA(delta_matrix[tr])  # apply saturating function to each time step

DA_avg = DA_matrix.mean(axis=0)

# # Now compare "avg_delta_all" vs. "DA_avg"
# plt.figure(figsize=(8,4))
# plt.plot(time_array, avg_delta_all, 'b-', label='Avg Delta(t)')
# plt.plot(time_array, DA_avg,        'r-', label='Avg DA( Delta(t) )')
# plt.title('Saturating Dopamine Signal vs. Raw Delta (Last 100 Trials)')
# plt.xlabel('Time (s)')
# plt.ylabel('Signal')
# plt.legend()
# plt.tight_layout()
# plt.show()


##############################################################################
# QUESTION 6 (Different Reward Probability)
##############################################################################

# We will run multiple "experiments," each with a different reward probability p
p_values = [0.0, 0.25, 0.5, 0.75, 1.0]

# Dictionary for storing the final (last 100 trials) average DA(t) curve, keyed by p
da_curves = {}

for p_reward in p_values:
    
    # ========== Initialize weights for each new experiment ========== #
    w = np.zeros(memory_steps)
    
    # For storing the TD error across all trials/time
    delta_matrix = np.zeros((num_trials, time_steps))
    
    # We also track which trials are rewarded
    is_rewarded = np.zeros(num_trials, dtype=bool)
    
    # ========== Main loop ========== #
    for trial in range(num_trials):
        # Decide if reward occurs this trial
        rewarded = (np.random.rand() < p_reward)
        is_rewarded[trial] = rewarded
        
        V = np.zeros(time_steps)  # track value for this trial

        # V(0)
        phi_0 = boxcar_feature_vector(0)
        V[0]  = np.dot(w, phi_0)
        
        for n in range(1, time_steps):
            phi_n = boxcar_feature_vector(n)
            V[n]  = np.dot(w, phi_n)
            
            reward_val = r((n-1)*dt) if rewarded else 0.0
            td_error   = reward_val + gamma*V[n] - V[n-1]
            
            # Weight update
            phi_prev = boxcar_feature_vector(n-1)
            w += epsilon * td_error * phi_prev
        
        # Compute delta(t) for logging
        delta_arr = np.zeros(time_steps)
        # For n=0
        delta_arr[0] = gamma*V[0]  # minus V(-1)=0 => delta(0) = V(0)
        
        for n in range(1, time_steps):
            r_prev      = r((n-1)*dt) if rewarded else 0.0
            delta_arr[n] = r_prev + gamma*V[n] - V[n-1]
        
        delta_matrix[trial] = delta_arr
    
    # ========== After all trials, compute average DA(t) in last 100 trials ========== #
    last_100 = np.arange(num_trials - 100, num_trials)
    
    # We'll compute DA for each trial's delta(t), then average
    DA_matrix = np.zeros_like(delta_matrix[last_100])
    for i, tr_idx in enumerate(last_100):
        DA_matrix[i] = DA(delta_matrix[tr_idx])
    
    DA_avg = DA_matrix.mean(axis=0)  # average over time in last 100 trials
    da_curves[p_reward] = DA_avg

# # ========== Plot the different p curves in one figure ==========
# plt.figure(figsize=(8,5))
# cmap = plt.cm.viridis(np.linspace(0,1,len(p_values)))

# for i, p in enumerate(p_values):
#     plt.plot(time_array, da_curves[p], color=cmap[i], label=f'p={p}')
    
# plt.title('Average Dopamine Time Course (last 100 trials) for Different p')
# plt.xlabel('Time (s)')
# plt.ylabel('DA( δ(t) )')
# plt.legend()
# plt.tight_layout()
# plt.show()


##############################################################################
# QUESTION 7 (Average Level of Dopamine)
##############################################################################


stimulus_idx = int(10.0 / dt)    # around 10 s
reward_idx   = int(20.0 / dt)    # around 20 s

peak_stim_list = []
peak_reward_list = []

for p in p_values:
    da_array = da_curves[p]
    
    # Extract a small window around each time, e.g. ±2 steps, to find the local max
    # (You can adjust window size as needed.)
    window_size = 2
    
    # bounds for the stimulus window
    stim_start = max(stimulus_idx - window_size, 0)
    stim_end   = min(stimulus_idx + window_size + 1, time_steps)
    peak_stim  = np.max(da_array[stim_start:stim_end])
    
    # bounds for the reward window
    rew_start  = max(reward_idx - window_size, 0)
    rew_end    = min(reward_idx + window_size + 1, time_steps)
    peak_rew   = np.max(da_array[rew_start:rew_end])
    
    peak_stim_list.append(peak_stim)
    peak_reward_list.append(peak_rew)

# # Now we can plot these peak values vs. p
# plt.figure(figsize=(6,4))
# plt.plot(p_values, peak_stim_list, 'o--', label='Peak DA near Stimulus (t=10s)')
# plt.plot(p_values, peak_reward_list,'o--', label='Peak DA near Reward (t=20s)')
# plt.xlabel('Reward Probability p')
# plt.ylabel('Peak DA signal')
# plt.title('Peak Dopamine at Stimulus vs. Reward Times')
# plt.legend()
# plt.tight_layout()
# plt.show()









