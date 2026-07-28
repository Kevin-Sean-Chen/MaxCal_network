"""CTMC and leaky integrate-and-fire simulation helpers."""

import numpy as np

def sim_Q(Q, total_time, time_step):
    """
    Simulate Markov chain given rate matrix Q, time length and steps
    reading this: https://www.columbia.edu/~ww2040/6711F13/CTMCnotes120413.pdf
    """
    nc = Q.shape[0]
    initial_state = np.random.randint(nc) # uniform to begin with
    states = [initial_state]
    times = [0.0]

    current_state = initial_state
    current_time = 0.0

    while current_time < total_time:
        rate = -Q[current_state, current_state]
        next_time = current_time + np.random.exponential(scale=1/rate)
        if next_time > total_time:
            break

        transition_probabilities = Q[current_state,:]*1#expm(Q * (next_time - current_time))[current_state, :]
        transition_probabilities[current_state] = 0  # remove diagonal
        transition_probabilities /= transition_probabilities.sum()  # Normalize probabilities
        next_state = np.random.choice(len(Q), p=transition_probabilities)
        # print(transition_probabilities)
        
        #####
        # # Generate exponentially distributed time until the next event 
        # rate = abs(Q[current_state, current_state]) 
        # time_to_next_event = np.random.exponential(scale=1/rate) # Update the time and state 
        # time_points.append(time_points[-1] + time_to_next_event) # Determine the next state based on transition probabilities 
        # transition_probs = Q[current_state, :] / rate transition_probs[transition_probs < 0] = 0  # Ensure non-negative probabilities 
        # transition_probs /= np.sum(transition_probs)  # Normalize probabilities to sum to 1 
        # next_state = np.random.choice(len(Q), p=transition_probs) 
        # state_sequence.append(next_state)
        #####
        
        states.append(next_state)
        times.append(next_time)

        current_state = next_state
        current_time = next_time

    return np.array(states), np.array(times)

# %% LIF model
def LIF_firing(lt):
    """
    given synaptic weights and noise amplitude, turn 3-neuron spiking time series
    """
    dt = 0.1  # time step in milliseconds
    timesteps = lt*1  #30000  # total simulation steps

    # Neuron parameters
    tau = 10.0  # membrane time constant
    v_rest = -65.0  # resting membrane potential
    v_threshold = -50.0  # spike threshold
    v_reset = -65.0  # reset potential after a spike

    # Synaptic weight matrix
    synaptic_weights = np.array([[0, 1, -2],  # Neuron 0 connections
                                  [1, 0, -2],  # Neuron 1 connections
                                  [1, 1, 0]])*20  #20  # Neuron 2 connections
    # synaptic_weights = (np.random.rand(3,3)+1)*20
    # sign = np.random.randn(3,3); sign[sign>0]=1; sign[sign<0] = -1
    # synaptic_weights = synaptic_weights*sign
    S = synaptic_weights*1
    np.fill_diagonal(S, np.zeros(3))
    noise_amp = 2

    # Synaptic filtering parameters
    tau_synaptic = 5.0  # synaptic time constant

    # Initialize neuron membrane potentials and synaptic inputs
    v_neurons = np.zeros((3, timesteps))
    synaptic_inputs = np.zeros((3, timesteps))
    spike_times = []
    firing = []
    firing.append((np.array([]), np.array([]))) # init

    # Simulation loop
    for t in range(1, timesteps):

        # Update neuron membrane potentials using leaky integrate-and-fire model
        v_neurons[:, t] = v_neurons[:, t - 1] + dt/tau*(v_rest - v_neurons[:, t - 1]) + np.random.randn(3)*noise_amp
        
        # Check for spikes
        spike_indices = np.where(v_neurons[:, t] > v_threshold)[0]
        
        # Apply synaptic connections with synaptic filtering
        synaptic_inputs[:, t] = synaptic_inputs[:, t-1] + dt*( \
                                -synaptic_inputs[:, t-1]/tau_synaptic + np.sum(synaptic_weights[:, spike_indices], axis=1))
        # synaptic_inputs[:, t] = np.sum(synaptic_weights[:, spike_indices], axis=1)
        # Update membrane potentials with synaptic inputs
        v_neurons[:, t] += synaptic_inputs[:, t]*dt
        
        # record firing
        firing.append([t+0*spike_indices, spike_indices])
        
        # reset and record spikes
        v_neurons[spike_indices, t] = v_reset  # Reset membrane potential for neurons that spiked
        if len(spike_indices) > 0:
            spike_times.append(t * dt)
    
    return firing

