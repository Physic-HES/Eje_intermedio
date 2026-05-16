import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

# 1. Parámetros y Ecuaciones de Euler (Sistema del Cuerpo)
I1, I2, I3 = 21.65, 17.71, 6.76

def euler_eom(t, omega):
    w1, w2, w3 = omega
    dw1_dt = ((I2 - I3) / I1) * w2 * w3
    dw2_dt = ((I3 - I1) / I2) * w3 * w1
    dw3_dt = ((I1 - I2) / I3) * w1 * w2
    return [dw1_dt, dw2_dt, dw3_dt]

# Simulación de la velocidad angular (Paso 1)
omega0 = [0.0001, 1.0, 0.0]
t_span = (0, 185)
t_eval = np.linspace(t_span[0], t_span[1], 1000)
sol_omega = solve_ivp(euler_eom, t_span, omega0, t_eval=t_eval, method='DOP853', dense_output=True, rtol=1e-10, atol=1e-12)

# 2. Integración de la Orientación (Paso 2)
def orientation_eom(t, R_flat):
    R = R_flat.reshape((3, 3))
    w_body = sol_omega.sol(t)
    w1, w2, w3 = w_body
    Omega = np.array([[ 0, -w3,  w2],
                      [ w3,  0, -w1],
                      [-w2,  w1,  0]])
    dRdt = R @ Omega
    return dRdt.flatten()

R0 = np.eye(3).flatten()
sol_R = solve_ivp(orientation_eom, t_span, R0, t_eval=t_eval, method='DOP853', rtol=1e-10, atol=1e-12)

# 3. Post-procesamiento
R_t = sol_R.y.T.reshape(-1, 3, 3)
w_body_t = sol_omega.y.T
w_fixed_t = np.array([R_t[i] @ w_body_t[i] for i in range(len(t_eval))])

# 4. Generación de Figuras Estáticas (para el Artículo)
# Función auxiliar para configurar el plot 3D
def setup_3d_axes(ax):
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xlabel('X Fijo')
    ax.set_ylabel('Y Fijo')
    ax.set_zlabel('Z Fijo')

# A. Guardar Frame 3D Final
fig_3d = plt.figure(figsize=(8, 8))
ax3d_final = fig_3d.add_subplot(111, projection='3d')
setup_3d_axes(ax3d_final)
R_final = R_t[-1]
colors = ['r', 'g', 'b']
for i in range(3):
    e_i = R_final[:, i]
    ax3d_final.plot([0, e_i[0]], [0, e_i[1]], [0, e_i[2]], color=colors[i], lw=2, label=f'Eje {i+1}')
wf_final = w_fixed_t[-1]
ax3d_final.plot([0, wf_final[0]], [0, wf_final[1]], [0, wf_final[2]], color='black', lw=3, label=r'$\vec{\omega}$')
trace_data = R_t[:, :, 1]
ax3d_final.plot(trace_data[:, 0], trace_data[:, 1], trace_data[:, 2], color='gray', lw=0.5, alpha=0.5)
ax3d_final.set_title('Estado Final: Orientación 3D')
ax3d_final.legend()
fig_3d.savefig('figura_3d.png')
plt.close(fig_3d)

# B. Guardar Evolución Sistema Propio
plt.figure(figsize=(8, 5))
plt.plot(t_eval, w_body_t[:, 0], 'r', label=r'$\omega_1$')
plt.plot(t_eval, w_body_t[:, 1], 'g', label=r'$\omega_2$')
plt.plot(t_eval, w_body_t[:, 2], 'b', label=r'$\omega_3$')
plt.title(r'Evolución Angular en el Sistema Propio ($\omega_{1,2,3}$)')
plt.xlabel('Tiempo [s]')
plt.ylabel(r'$\omega$ [rad/s]')
plt.legend()
plt.grid(True)
plt.savefig('figura_omega_cuerpo.png')
plt.close()

# C. Guardar Evolución Sistema Fijo
plt.figure(figsize=(8, 5))
plt.plot(t_eval, w_fixed_t[:, 0], label=r'$\omega_x$')
plt.plot(t_eval, w_fixed_t[:, 1], label=r'$\omega_y$')
plt.plot(t_eval, w_fixed_t[:, 2], label=r'$\omega_z$')
plt.title(r'Evolución Angular en el Sistema Fijo ($\omega_{x,y,z}$)')
plt.xlabel('Tiempo [s]')
plt.ylabel(r'$\omega$ [rad/s]')
plt.legend()
plt.grid(True)
plt.savefig('figura_omega_fijo.png')
plt.close()

# 5. Animación Interactiva (Layout original 16:9 sin el gráfico de error)
fig = plt.figure(figsize=(16, 9))
gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 1])

ax3d = fig.add_subplot(gs[:, 0], projection='3d')
setup_3d_axes(ax3d)
ax3d.set_title('Visualización 3D: Ejes Principales')

ax_body = fig.add_subplot(gs[0, 1])
ax_body.set_xlim(t_span)
ax_body.set_ylim([np.min(w_body_t)*1.2, np.max(w_body_t)*1.2])
ax_body.set_title(r'Sistema Propio ($\omega_{1,2,3}$)')
ax_body.grid(True)

ax_fixed = fig.add_subplot(gs[1, 1])
ax_fixed.set_xlim(t_span)
ax_fixed.set_ylim([np.min(w_fixed_t)*1.2, np.max(w_fixed_t)*1.2])
ax_fixed.set_title(r'Sistema Fijo ($\omega_{x,y,z}$)')
ax_fixed.set_xlabel('Tiempo [s]')
ax_fixed.grid(True)

lines3d = [ax3d.plot([], [], [], color=colors[i], lw=2, label=f'Eje {i+1}')[0] for i in range(3)]
omega_line3d, = ax3d.plot([], [], [], color='black', lw=3, label=r'$\vec{\omega}$')
trace3d, = ax3d.plot([], [], [], color='gray', lw=0.5, alpha=0.5)

lines_body = [ax_body.plot([], [], color=colors[i], label=rf'$\omega_{i+1}$')[0] for i in range(3)]
lines_fixed = [ax_fixed.plot([], [], label=['x','y','z'][i])[0] for i in range(3)]

def update(num):
    R = R_t[num]; wf = w_fixed_t[num]
    for i in range(3):
        e_i = R[:, i]
        lines3d[i].set_data([0, e_i[0]], [0, e_i[1]])
        lines3d[i].set_3d_properties([0, e_i[2]])
    omega_line3d.set_data([0, wf[0]], [0, wf[1]])
    omega_line3d.set_3d_properties([0, wf[2]])
    trace3d.set_data(trace_data[:num, 0], trace_data[:num, 1])
    trace3d.set_3d_properties(trace_data[:num, 2])
    
    for i in range(3):
        lines_body[i].set_data(t_eval[:num], w_body_t[:num, i])
        lines_fixed[i].set_data(t_eval[:num], w_fixed_t[:num, i])
    
    return lines3d + [omega_line3d, trace3d] + lines_body + lines_fixed

ani = FuncAnimation(fig, update, frames=len(t_eval), interval=20, blit=True)
ax_body.legend(loc='upper right', fontsize='small')
ax_fixed.legend(loc='upper right', fontsize='small')
ax3d.legend(loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()
