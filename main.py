import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time

def mandelbrot_gpu(width, height, x_min, x_max, y_min, y_max, max_iter, c_custom, power):
    x, y = cp.linspace(x_min, x_max, width), cp.linspace(y_min, y_max, height)
    X, Y = cp.meshgrid(x, y)
    z = X + 1j * Y
    c = cp.full(z.shape, c_custom, dtype=complex)
    iteration_counts = cp.zeros(z.shape, dtype=int)
    mask = cp.ones(z.shape, dtype=bool)

    for i in range(max_iter):
        z[mask] = z[mask]**power + c[mask]
        mask = mask & (cp.abs(z) <= 2)
        iteration_counts[mask] += 1

    return cp.asnumpy(iteration_counts)

width, height = 400, 400  
x_min, x_max, y_min, y_max = -2.0, 2.0, -2.0, 2.0
max_iter = 100
initial_c = 0.355 + 0.355j
initial_power = 2.0

fig, ax = plt.subplots(figsize=(10, 10))
plt.subplots_adjust(left=0.25, bottom=0.25)

mandelbrot_image = mandelbrot_gpu(width, height, x_min, x_max, y_min, y_max, max_iter, initial_c, initial_power)
im = ax.imshow(mandelbrot_image, extent=[x_min, x_max, y_min, y_max], cmap='hot', origin='lower')
ax.set_title(f"Interactive Mandelbrot Set with GPU Acceleration")
ax.set_xlabel("Re(z)")
ax.set_ylabel("Im(z)")

axcolor = 'lightgoldenrodyellow'
ax_c_real = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_c_imag = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_power = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
ax_zoom = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
ax_iter = plt.axes([0.25, 0.3, 0.65, 0.03], facecolor=axcolor)

slider_c_real = Slider(ax_c_real, 'Re(c)', -2.0, 2.0, valinit=initial_c.real)
slider_c_imag = Slider(ax_c_imag, 'Im(c)', -2.0, 2.0, valinit=initial_c.imag)
slider_power = Slider(ax_power, 'Power (z^x)', 1.0, 4.0, valinit=initial_power)
slider_zoom = Slider(ax_zoom, 'Zoom', 0.5, 2.0, valinit=1.0)
slider_iter = Slider(ax_iter, 'Max Iter', 50, 500, valinit=max_iter)

last_update_time = 0
debounce_interval = 0.2  

def update(val):
    global last_update_time
    
    current_time = time.time()
    if current_time - last_update_time < debounce_interval:
        return  
 
    c_custom = complex(slider_c_real.val, slider_c_imag.val)
    power = slider_power.val
    zoom = slider_zoom.val
    max_iter = int(slider_iter.val)
    
    x_range = (x_max - x_min) * zoom
    y_range = (y_max - y_min) * zoom
    x_min_adjusted = initial_c.real - x_range / 2
    x_max_adjusted = initial_c.real + x_range / 2
    y_min_adjusted = initial_c.imag - y_range / 2
    y_max_adjusted = initial_c.imag + y_range / 2
    
    mandelbrot_image = mandelbrot_gpu(width, height, x_min_adjusted, x_max_adjusted, y_min_adjusted, y_max_adjusted, max_iter, c_custom, power)
    
    im.set_data(mandelbrot_image)
    ax.set_title(f"Power = {power}, c = {c_custom}, Max Iter = {max_iter}, Zoom = {zoom:.2f}")
    fig.canvas.draw_idle()
    
    last_update_time = current_time

slider_c_real.on_changed(update)
slider_c_imag.on_changed(update)
slider_power.on_changed(update)
slider_zoom.on_changed(update)
slider_iter.on_changed(update)

plt.show()
