import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons


def generate_rect_pulse(t, width, center=0.5):
    return ((t >= (center - width/2)) & (t < (center + width/2))).astype(float)

def generate_step(t, t0=0.0):
    return (t >= t0).astype(float)

def generate_sine(t, f, A=1.0, phase=0.0):
    return A * np.sin(2*np.pi*f*t + phase)

def compute_fft(x, fs):
    N = len(x)
    X = np.fft.fft(x)
    X_shift = np.fft.fftshift(X) / N
    freqs = np.fft.fftfreq(N, d=1/fs)
    freqs_shift = np.fft.fftshift(freqs)
    return freqs_shift, X_shift


fs = 2000.0
T = 1.0
t = np.arange(0, T, 1/fs)


signal_type = "Seno"
f_init, A_init, phase_init = 5.0, 1.0, 0.0
signal = generate_sine(t, f_init, A_init, phase_init)
freqs, X = compute_fft(signal, fs)


fig, (ax_time, ax_mag, ax_phase) = plt.subplots(3, 1, figsize=(8, 6))
plt.subplots_adjust(left=0.25, bottom=0.35)


line_time, = ax_time.plot(t, signal)
ax_time.set_title("Señal en el tiempo")
ax_time.set_xlim(0, 1)

line_mag, = ax_mag.plot(freqs, np.abs(X))
ax_mag.set_title("Magnitud FFT")

line_phase, = ax_phase.plot(freqs, np.unwrap(np.angle(X)))
ax_phase.set_title("Fase FFT")

for ax in (ax_time, ax_mag, ax_phase):
    ax.grid(True)


ax_freq = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_amp = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_phase_slider = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_width = plt.axes([0.25, 0.10, 0.65, 0.03])

s_freq = Slider(ax_freq, 'Frecuencia', 1, 50, valinit=f_init)
s_amp = Slider(ax_amp, 'Amplitud', 0.1, 2.0, valinit=A_init)
s_phase = Slider(ax_phase_slider, 'Fase', -np.pi, np.pi, valinit=phase_init)
s_width = Slider(ax_width, 'Ancho Pulso', 0.01, 0.5, valinit=0.2)


rax = plt.axes([0.025, 0.5, 0.15, 0.3])
radio = RadioButtons(rax, ('Seno', 'Pulso', 'Escalón'))


def update(val):
    global signal_type
    if signal_type == "Seno":
        f = s_freq.val
        A = s_amp.val
        phase = s_phase.val
        sig = generate_sine(t, f, A, phase)
    elif signal_type == "Pulso":
        w = s_width.val
        sig = generate_rect_pulse(t, width=w, center=0.5)
    elif signal_type == "Escalón":
        sig = generate_step(t, t0=0.3)

    freqs, X = compute_fft(sig, fs)


    line_time.set_ydata(sig)
    line_mag.set_ydata(np.abs(X))
    line_phase.set_ydata(np.unwrap(np.angle(X)))

    fig.canvas.draw_idle()

def change_signal(label):
    global signal_type
    signal_type = label
    update(None)

s_freq.on_changed(update)
s_amp.on_changed(update)
s_phase.on_changed(update)
s_width.on_changed(update)
radio.on_clicked(change_signal)

plt.show()
