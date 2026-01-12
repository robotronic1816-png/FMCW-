import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows, butter, filtfilt

# CFAR FUNCTION
def cfar_2d(rd_map,
            num_train_r=10, num_guard_r=4,
            num_train_d=6,  num_guard_d=2,
            threshold_scale=5.5):

    detections = np.zeros_like(rd_map, dtype=int)
    D, R = rd_map.shape

    for d in range(num_train_d + num_guard_d, D - (num_train_d + num_guard_d)):
        for r in range(num_train_r + num_guard_r, R - (num_train_r + num_guard_r)):

            noise_cells = []

            for dd in range(d - num_train_d - num_guard_d, d + num_train_d + num_guard_d + 1):
                for rr in range(r - num_train_r - num_guard_r, r + num_train_r + num_guard_r + 1):
                    if abs(dd - d) <= num_guard_d and abs(rr - r) <= num_guard_r:
                        continue
                    noise_cells.append(rd_map[dd, rr])

            threshold = np.mean(noise_cells) + threshold_scale
            if rd_map[d, r] > threshold:
                detections[d, r] = 1

    return detections


# RADAR PARAMETERS 
c = 3e8
fc = 77e9
B = 200e6
T = 40e-6
S = B / T
fs = 20e6

R = 50.0
v = -10.0
num_chirps = 64
SNR_dB = 15
# TIME AXIS
t = np.arange(0, T, 1/fs)
# TRANSMIT IQ CHIRP
tx = np.exp(1j * 2 * np.pi * (S / 2) * t**2)

# Doppler frequency
f_D = 2 * v * fc / c

# IF FILTER
b, a = butter(4, 0.25)
# BEAT MATRIX (IQ)
beat_matrix = np.zeros((num_chirps, len(t)), dtype=complex)

for k in range(num_chirps):

    R_k = R + v * k * T
    tau_k = 2 * R_k / c

    rx_k = np.exp(
        1j * (
            2 * np.pi * (S / 2) * (t - tau_k)**2
            + 2 * np.pi * f_D * (k * T + t)
        )
    )

    # Dechirp
    beat = tx * np.conj(rx_k)

    # Add noise
    signal_power = np.mean(np.abs(beat)**2)
    noise_power = signal_power / (10**(SNR_dB / 10))

    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(t)) + 1j * np.random.randn(len(t))
    )

    beat = beat + noise

    # IF filtering
    beat = filtfilt(b, a, beat)

    beat_matrix[k, :] = beat


# RANGE FFT
beat_matrix *= windows.hann(len(t))[None, :]

range_fft = np.fft.fft(beat_matrix, axis=1)
range_fft = range_fft[:, :len(t)//2]

freq_r = np.fft.fftfreq(len(t), d=1/fs)[:len(t)//2]
ranges = (c * freq_r) / (2 * S)
# DOPPLER FFT
range_fft *= windows.hann(num_chirps)[:, None]

doppler_fft = np.fft.fftshift(
    np.fft.fft(range_fft, axis=0),
    axes=0
)

rd_map = np.abs(doppler_fft)
# AXES
doppler_freq = np.fft.fftshift(np.fft.fftfreq(num_chirps, d=T))
velocity_axis = doppler_freq * (c / (2 * fc))
# CFAR
rd_map_db = 20 * np.log10(rd_map + 1e-6)

detections = cfar_2d(rd_map_db)
# PEAK PICKING
det_d, det_r = np.where(detections == 1)

if len(det_r) > 0:
    powers = rd_map_db[det_d, det_r]
    idx = np.argmax(powers)

    est_range = ranges[det_r[idx]]
    est_velocity = velocity_axis[det_d[idx]]

    print(" FINAL TARGET ESTIMATE ")
    print(f"Range    : {est_range:.2f} m")
    print(f"Velocity : {est_velocity:.2f} m/s")
else:
    print("No target detected.")
# PLOT
plt.figure(figsize=(9,6))
plt.imshow(
    rd_map_db,
    aspect="auto",
    extent=[0, np.max(ranges),
            velocity_axis[0], velocity_axis[-1]],
    origin="lower",
    cmap="jet"
)

plt.scatter(
    ranges[det_r],
    velocity_axis[det_d],
    c="red", s=12
)

plt.xlabel("Distance (m)")
plt.ylabel("Velocity (m/s)")
plt.title("Range–Doppler Map with CFAR Detections")
plt.colorbar(label="Amplitude (dB)")
plt.tight_layout()
plt.show()
