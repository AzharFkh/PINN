import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np 

# GPU Nvidia config
gpus = tf.config.experimental.list_physical_devices('GPU')
memory = 2048

if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit= memory)]
        )
        print(f"GPU memory limiting to {memory} MB")
    except RuntimeError as e:
        print(e)

# Exact solution from equation
delta = 2
w0 = 20
mu = 2*delta
k = w0**2
t_Test = tf.reshape(np.linspace(0, 1, 300), (-1, 1))

def exact_solution(d, w0, t):
    w = np.sqrt(w0**2 - d**2)
    phi = np.arctan(-d/w)
    A = 1/(2*np.cos(phi))
    cos = tf.cos(phi+w*t)
    exp = tf.exp(-d*t)
    u = exp*2*A*cos
    return u

u_exact = exact_solution(d=delta, w0=w0, t=t_Test)

# PINNs architecture 
tf.random.set_seed(42)

class PINN(tf.keras.Model):
    def __init__(self, Neuron_input, Neuorn_hidden, Neuron_output, activation = 'tanh'):
        super().__init__()

        self.input_layer = tf.keras.layers.Dense(Neuorn_hidden, activation=activation, input_shape=(Neuron_input,))
        self.hidden_layer = tf.keras.layers.Dense(Neuorn_hidden, activation=activation)
        self.output_layer = tf.keras.layers.Dense(Neuron_output)

    def call(self, x):               # Feed forward propagation
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        x = self.hidden_layer(x)    #2nd hidden layer
        x = self.output_layer(x)
        return x 
    
pinn_solution = PINN(1, 32, 1)

# training data & loss calculation 
optimizer_pinn = tf.keras.optimizers.Adam(learning_rate=1e-3)
t_boundary = tf.Variable([[0.0]], dtype=tf.float32)
t_physics = tf.Variable(tf.linspace(0.0, 1.0, 30)[:, tf.newaxis], dtype=tf.float32)
lambda1, lambda2 = 1e-1, 1e-3
# ini pake yang pinn_solution
history_loss = [] # Menggunakan satu history loss saja untuk total loss

for i in range(20_001):
    # Backpropagation
    with tf.GradientTape() as tape_outer:

        # Boundary Loss
        with tf.GradientTape() as tape_boundary:
            tape_boundary.watch(t_boundary)
            u_boundary = pinn_solution(t_boundary)
        
        dudt_boundary = tape_boundary.gradient(u_boundary, t_boundary)
        del tape_boundary 

        loss1 = tf.reduce_mean(tf.square(u_boundary - 1.0))
        loss2 = tf.reduce_mean(tf.square(dudt_boundary - 0.0))


        # Physics-Informed Loss 
        with tf.GradientTape(persistent=True) as tape_physics:
            tape_physics.watch(t_physics)
            u_physics = pinn_solution(t_physics)
            dudt_physics = tape_physics.gradient(u_physics, t_physics)
    
        d2udt2_physics = tape_physics.gradient(dudt_physics, t_physics)
        del tape_physics # Hapus tape dalam setelah selesai

        # Physics Residual Loss
        physics_residual = d2udt2_physics + mu * dudt_physics + k * u_physics
        loss3 = tf.reduce_mean(tf.square(physics_residual))


        # Total Loss
        total_loss = loss1 + lambda1 * loss2 + lambda2 * loss3
    
    # ================== AKHIR DARI TAPE LUAR ==================

    # Backpropagation: Hitung gradien loss total terhadap parameter model.
    gradients = tape_outer.gradient(total_loss, pinn_solution.trainable_variables)
    optimizer_pinn.apply_gradients(zip(gradients, pinn_solution.trainable_variables))

    history_loss.append(total_loss.numpy())

    if i % 1000 == 0:
        print(f"Iter {i}: Loss = {total_loss.numpy():.5f}")

u = pinn_solution(t_Test)
u_detached = tf.stop_gradient(u)
plt.figure(figsize=(8,4))
plt.plot(t_Test[:,0], u_exact[:,0], label='Excact Solution', color='tab:red')
plt.plot(t_Test[:,0], u_detached[:,0], label='PINN Solution', color='tab:blue')
plt.title('Damped Oscilation Har')
plt.legend()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(history_loss)
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Training Loss')
plt.grid(True)
plt.show()