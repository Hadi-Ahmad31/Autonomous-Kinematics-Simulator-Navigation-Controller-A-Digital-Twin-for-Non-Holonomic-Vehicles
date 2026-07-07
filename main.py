import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math



r = 0.05  
d = 0.20  
dt = 0.1  


waypoints = [(1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
current_wp_index = 0  


Kp_linear = 1.0

Kp_angular = 4.0
Ki_angular = 0.01
Kd_angular = 1.5


x, y, theta = 0.0, 0.0, 0.0  
est_x, est_y = 0.0, 0.0  
x_history, y_history = [], []
noisy_x_hist, noisy_y_hist = [], [] 

# Variables to store historical data for the I and D terms
prev_heading_error = 0.0
integral_heading_error = 0.0


fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Final: Full PID Control with Filtered Kinematics")

wp_x = [wp[0] for wp in waypoints]
wp_y = [wp[1] for wp in waypoints]
ax.plot(wp_x, wp_y, 'g*', markersize=15, alpha=0.3, label='Mission Path')

active_target_pt, = ax.plot([], [], 'g*', markersize=15, label='Active Target')
trail, = ax.plot([], [], 'b-', alpha=0.5, label='True Path')
sensor_scatter, = ax.plot([], [], 'k.', alpha=0.2, markersize=3, label='Noisy Sensor')
robot, = ax.plot([], [], 'ro', markersize=12, label='Robot (Truth)')
heading, = ax.plot([], [], 'k-', linewidth=2)
ax.legend()


def update(frame):
    global x, y, theta, current_wp_index, est_x, est_y
    global prev_heading_error, integral_heading_error
    prev_derivative = 0.0
    
    if current_wp_index >= len(waypoints):
        return trail, robot, heading, active_target_pt, sensor_scatter
        
    x_target, y_target = waypoints[current_wp_index]
    
   
    measured_x = x + np.random.normal(0, 0.03)
    measured_y = y + np.random.normal(0, 0.03)

    alpha = 0.2 
    est_x = (alpha * measured_x) + ((1 - alpha) * est_x)
    est_y = (alpha * measured_y) + ((1 - alpha) * est_y)
    
  
    distance_error = math.sqrt((x_target - est_x)**2 + (y_target - est_y)**2)
    
    if distance_error < 0.05:
        current_wp_index += 1
        # Reset the integral term when we switch to a new target to prevent "windup"
        integral_heading_error = 0.0 
        return trail, robot, heading, active_target_pt, sensor_scatter
        
    desired_theta = math.atan2(y_target - est_y, x_target - est_x)
    heading_error = desired_theta - theta
    heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

   
    P_out = Kp_angular * heading_error
    
   
    integral_heading_error += (heading_error * dt)
    I_out = Ki_angular * integral_heading_error
    
    
    delta_error = heading_error - prev_heading_error
    delta_error = math.atan2(math.sin(delta_error), math.cos(delta_error))
    raw_derivative = delta_error / dt
    
    
    filtered_derivative = (0.1 * raw_derivative) + (0.9 * prev_derivative)
    prev_derivative = filtered_derivative
    
    D_out = Kd_angular * filtered_derivative
    
    
    omega = P_out + I_out + D_out
    prev_heading_error = heading_error
    
    
    v = Kp_linear * distance_error
    
    
    if abs(heading_error) > 0.26:
        v = 0.0 
    
   
    v = max(min(v, 0.5), -0.5) 
    omega = max(min(omega, 2.0), -2.0)

   
    w_R = (v + (omega * d / 2.0)) / r
    w_L = (v - (omega * d / 2.0)) / r

    x += v * math.cos(theta) * dt
    y += v * math.sin(theta) * dt
    theta += omega * dt

    
    x_history.append(x)
    y_history.append(y)
    noisy_x_hist.append(measured_x)
    noisy_y_hist.append(measured_y)
    
    if len(noisy_x_hist) > 150:
        noisy_x_hist.pop(0)
        noisy_y_hist.pop(0)
    
    trail.set_data(x_history, y_history)
    robot.set_data([x], [y])
    sensor_scatter.set_data(noisy_x_hist, noisy_y_hist)
    heading.set_data([x, x + 0.15 * math.cos(theta)], [y, y + 0.15 * math.sin(theta)])
    active_target_pt.set_data([x_target], [y_target])
                     
    return trail, robot, heading, active_target_pt, sensor_scatter

ani = animation.FuncAnimation(fig, update, frames=500, interval=50, blit=True)
plt.show()