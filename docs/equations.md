# Electrical equations and signal-processing decisions

This document records the mathematical conventions used by the application. The implementation analyzes one synchronized voltage/current channel and reports SI units.

## RMS values

For a sampled signal `x[n]` with `N` samples, true RMS is

```text
x_rms = sqrt((1/N) * sum(x[n]^2))
```

The complete waveform is used, so DC offset, noise, and harmonic content all contribute. No sinusoidal assumption is required.

## Active, apparent, and reactive power

Instantaneous and active power are

```text
p[n] = v[n] i[n]
P = (1/N) * sum(p[n])
```

Apparent power and true power factor are

```text
S = V_rms I_rms
PF = P / S
```

These definitions remain valid for distorted periodic signals. The power factor therefore includes both phase displacement and waveform distortion.

There is no single universally interchangeable scalar `Q` for arbitrary non-sinusoidal waveforms. This project reports **fundamental reactive power** `Q1`. It fits the voltage and current fundamental components by least squares, represents their RMS phasors as `V1` and `I1`, and evaluates

```text
S1_complex = V1 conjugate(I1)
P1 = real(S1_complex)
Q1 = imag(S1_complex)
```

With the adopted sign convention, lagging current produces positive `Q1`. The displayed displacement power factor is `P1 / (|V1||I1|)`. A residual non-active quantity is also available from

```text
N = sqrt(max(0, S^2 - P^2 - Q1^2))
```

`N` is diagnostic; it is not presented as a replacement definition for reactive power.

## Energy

Energy is the numerical time integral of instantaneous power:

```text
E_Wh = (1/3600) integral(v(t)i(t) dt)
```

The trapezoidal rule uses the supplied timestamps. A short capture yields a mathematically valid interval energy, but long-term billing energy requires a continuous, calibrated acquisition chain.

## FFT spectrum and fundamental frequency

Before the FFT, the mean is removed and a periodic Hann window is applied:

```text
X[k] = sum(x[n] w[n] exp(-j 2 pi k n / N))
```

The dashboard uses a one-sided spectrum. Interior positive-frequency bins are doubled to account for the omitted negative-frequency half. Dividing by `sum(w)` corrects the Hann window's coherent gain, so the result is a peak-amplitude spectrum for a bin-centered sinusoid.

The fundamental is the largest spectral component within the configured search interval (45–75 Hz by default). Zero padding and a three-point log-parabolic interpolation refine the displayed peak estimate. Zero padding does not create additional physical resolution; the acquisition duration still controls the ability to separate nearby tones.

## Harmonic magnitudes and THD

An FFT peak first establishes the fundamental frequency. Harmonic RMS magnitudes are then estimated together by linear least squares using sine/cosine bases at integer multiples of that frequency. This hybrid approach keeps the FFT-based spectral view while reducing scalloping and leakage bias when the capture is not cycle coherent.

For fundamental RMS magnitude `X1` and harmonic RMS magnitudes `Xh`, total harmonic distortion is

```text
THD = 100% * sqrt(sum(Xh^2, h=2..H)) / X1
```

The maximum order is configurable and capped below Nyquist. DC is excluded. The reported value therefore depends on sample rate, acquisition length, noise, and the configured maximum harmonic order.

## Event detection

Voltage magnitude is evaluated in non-overlapping, nominal one-cycle windows. Each window is assigned exactly one class, preventing a severe sag from also being counted as an undervoltage. Consecutive windows of the same class are merged; duration is the merged interval end minus its start.

Frequency and THD checks currently apply to the complete uploaded interval. The defaults in `src/config.py` are **demonstration thresholds**, not normative limits. Standards define measurement methods, aggregation, performance classes, and terminology; a production compliance tool would also require calibrated transducers, uncertainty analysis, prescribed synchronization/aggregation, and conformance testing.

