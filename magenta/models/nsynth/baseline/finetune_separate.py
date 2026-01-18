#!/usr/bin/env python
"""
Separate Fine-Tuning: Guitar-only and Bass-only models
Creates two separate fine-tuned models and comparison images
"""

import os
import sys
import glob
import json
import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set env before importing TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import tensorflow.compat.v1 as tf
import tf_slim as slim

tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()

# ============================================================================
# Configuration
# ============================================================================

WORKSPACE = "/workspaces/magenta-nsynth-du2projekt"
BASELINE_PATH = os.path.join(WORKSPACE, "magenta/models/nsynth/baseline")
GUITAR_DIR = os.path.join(BASELINE_PATH, "data/wavs/guitars")
BASS_DIR = os.path.join(BASELINE_PATH, "data/wavs/bass")
OUTPUT_DIR = os.path.join(WORKSPACE, "pipeline_results3")
BASELINE_CKPT = os.path.join(BASELINE_PATH, "baseline-ckpt/model.ckpt-351648")

SAMPLE_RATE = 16000
SAMPLE_LENGTH = 64000
N_FFT = 1024
HOP_LENGTH = 256
LEARNING_RATE = 1e-5
NUM_EPOCHS = 20
NUM_LATENT = 1984
SPEC_HEIGHT = 512
SPEC_WIDTH = 256

TEST_GUITAR = "ukelele - nice bro.wav"
TEST_BASS = "bass - guitarbass.wav"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Helper Functions
# ============================================================================

def leaky_relu(leak=0.1):
    return lambda x: tf.maximum(x, leak * x)

def slim_batchnorm_arg_scope(is_training, activation_fn=None):
    batch_norm_params = {
        "is_training": is_training,
        "decay": 0.999,
        "epsilon": 0.001,
        "variables_collections": {
            "beta": None, "gamma": None,
            "moving_mean": "moving_vars", "moving_variance": "moving_vars",
        }
    }
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected, slim.conv2d_transpose],
        weights_initializer=slim.initializers.xavier_initializer(),
        activation_fn=activation_fn,
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params) as scope:
        return scope

def conv2d(x, kernel_size, stride, channels, is_training, scope="conv2d",
           batch_norm=False, activation_fn=tf.nn.relu, transpose=False):
    conv_fn = slim.conv2d_transpose if transpose else slim.conv2d
    normalizer_fn = slim.batch_norm if batch_norm else None
    with tf.variable_scope(scope + "_Layer"):
        with slim.arg_scope(slim_batchnorm_arg_scope(is_training, activation_fn=None)):
            x = conv_fn(inputs=x, stride=stride, kernel_size=kernel_size,
                       num_outputs=channels, normalizer_fn=normalizer_fn,
                       biases_initializer=tf.zeros_initializer(), scope=scope)
            if activation_fn:
                x = activation_fn(x)
    return x

def pitch_embeddings(pitch, batch_size, n_pitches=128, dim_embedding=128):
    with tf.variable_scope("PitchEmbedding"):
        w = tf.get_variable(name="embedding_weights", shape=[n_pitches, dim_embedding],
                           initializer=tf.random_normal_initializer())
        one_hot_pitch = tf.one_hot(tf.reshape(pitch, [batch_size]), depth=n_pitches)
        embedding = tf.matmul(one_hot_pitch, w)
        embedding = tf.reshape(embedding, [batch_size, 1, 1, dim_embedding])
    return embedding

def specgram(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    f, t, Zxx = signal.stft(audio, fs=SAMPLE_RATE, nperseg=n_fft, noverlap=n_fft-hop_length)
    mag = np.abs(Zxx)
    mag = mag / (np.max(mag) + 1e-8)
    return mag.T

def ispecgram(mag, n_fft=N_FFT, hop_length=HOP_LENGTH, n_iter=50):
    mag = mag.T
    target_freq = n_fft // 2 + 1
    if mag.shape[0] < target_freq:
        mag = np.pad(mag, ((0, target_freq - mag.shape[0]), (0, 0)), mode='constant')
    elif mag.shape[0] > target_freq:
        mag = mag[:target_freq, :]
    angle = np.exp(2j * np.pi * np.random.rand(*mag.shape))
    S = mag * angle
    for _ in range(n_iter):
        _, audio = signal.istft(S, fs=SAMPLE_RATE, nperseg=n_fft, noverlap=n_fft-hop_length)
        _, _, S_new = signal.stft(audio, fs=SAMPLE_RATE, nperseg=n_fft, noverlap=n_fft-hop_length)
        angle = np.exp(1j * np.angle(S_new))
        S = mag * angle
    _, audio = signal.istft(S, fs=SAMPLE_RATE, nperseg=n_fft, noverlap=n_fft-hop_length)
    return audio.astype(np.float32)

def prepare_spectrogram(audio):
    spec = specgram(audio).T
    if spec.shape[0] > SPEC_HEIGHT:
        spec = spec[:SPEC_HEIGHT, :]
    elif spec.shape[0] < SPEC_HEIGHT:
        spec = np.pad(spec, ((0, SPEC_HEIGHT - spec.shape[0]), (0, 0)), mode='constant')
    if spec.shape[1] > SPEC_WIDTH:
        spec = spec[:, :SPEC_WIDTH]
    elif spec.shape[1] < SPEC_WIDTH:
        spec = np.pad(spec, ((0, 0), (0, SPEC_WIDTH - spec.shape[1])), mode='constant')
    return spec

def load_wav_file(path):
    try:
        audio, file_sr = sf.read(path)
        if file_sr != SAMPLE_RATE:
            audio = signal.resample(audio, int(len(audio) * SAMPLE_RATE / file_sr))
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        if len(audio) < SAMPLE_LENGTH:
            pad = SAMPLE_LENGTH - len(audio)
            audio = np.pad(audio, (pad // 2, pad - pad // 2), mode='constant')
        elif len(audio) > SAMPLE_LENGTH:
            start = (len(audio) - SAMPLE_LENGTH) // 2
            audio = audio[start:start + SAMPLE_LENGTH]
        return audio.astype(np.float32)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def compute_metrics(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    f_o, t_o, spec_o = signal.spectrogram(original, fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT-HOP_LENGTH)
    f_r, t_r, spec_r = signal.spectrogram(reconstructed, fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT-HOP_LENGTH)
    spec_o = spec_o / (np.max(np.abs(spec_o)) + 1e-8)
    spec_r = spec_r / (np.max(np.abs(spec_r)) + 1e-8)
    min_len = min(spec_o.shape[1], spec_r.shape[1])
    spec_dist = np.mean(np.sqrt(np.sum((spec_o[:, :min_len] - spec_r[:, :min_len]) ** 2, axis=0)))
    return {'mse': float(mse), 'spectral_dist': float(spec_dist)}

# ============================================================================
# Model Architecture
# ============================================================================

def encode(x, is_training):
    with tf.variable_scope("encoder"):
        h = conv2d(x, [5, 5], [2, 2], 128, is_training, activation_fn=leaky_relu(), batch_norm=True, scope="0")
        h = conv2d(h, [4, 4], [2, 2], 128, is_training, activation_fn=leaky_relu(), batch_norm=True, scope="1")
        h = conv2d(h, [4, 4], [2, 2], 128, is_training, activation_fn=leaky_relu(), batch_norm=True, scope="2")
        h = conv2d(h, [4, 4], [2, 2], 256, is_training, activation_fn=leaky_relu(), batch_norm=True, scope="3")
        h = conv2d(h, [4, 4], [2, 2], 256, is_training, activation_fn=leaky_relu(), batch_norm=True, scope="4")
        h = conv2d(h, [4, 4], [2, 2], 256, is_training, activation_fn=leaky_relu(), batch_norm=True, scope="5")
        h = conv2d(h, [4, 4], [2, 2], 512, is_training, activation_fn=leaky_relu(), batch_norm=True, scope="6")
        h = conv2d(h, [4, 4], [2, 2], 512, is_training, activation_fn=leaky_relu(), batch_norm=True, scope="7")
        h = conv2d(h, [4, 4], [2, 1], 512, is_training, activation_fn=leaky_relu(), batch_norm=True, scope="7_1")
        h = conv2d(h, [1, 1], [1, 1], 1024, is_training, activation_fn=leaky_relu(), batch_norm=True, scope="8")
        z = conv2d(h, [1, 1], [1, 1], NUM_LATENT, is_training, activation_fn=None, batch_norm=True, scope="z")
    return z

def decode(z, pitch, batch_size, is_training):
    with tf.variable_scope("decoder"):
        z_pitch = pitch_embeddings(pitch, batch_size)
        z = tf.concat([z, z_pitch], 3)
        h = conv2d(z, [1, 1], [1, 1], 1024, is_training, activation_fn=leaky_relu(), batch_norm=True, transpose=True, scope="0")
        h = conv2d(h, [4, 4], [2, 2], 512, is_training, activation_fn=leaky_relu(), batch_norm=True, transpose=True, scope="1")
        h = conv2d(h, [4, 4], [2, 2], 512, is_training, activation_fn=leaky_relu(), batch_norm=True, transpose=True, scope="2")
        h = conv2d(h, [4, 4], [2, 2], 256, is_training, activation_fn=leaky_relu(), batch_norm=True, transpose=True, scope="3")
        h = conv2d(h, [4, 4], [2, 2], 256, is_training, activation_fn=leaky_relu(), batch_norm=True, transpose=True, scope="4")
        h = conv2d(h, [4, 4], [2, 2], 256, is_training, activation_fn=leaky_relu(), batch_norm=True, transpose=True, scope="5")
        h = conv2d(h, [4, 4], [2, 2], 128, is_training, activation_fn=leaky_relu(), batch_norm=True, transpose=True, scope="6")
        h = conv2d(h, [4, 4], [2, 2], 128, is_training, activation_fn=leaky_relu(), batch_norm=True, transpose=True, scope="7")
        h = conv2d(h, [5, 5], [2, 2], 128, is_training, activation_fn=leaky_relu(), batch_norm=True, transpose=True, scope="8")
        h = conv2d(h, [5, 5], [2, 1], 128, is_training, activation_fn=leaky_relu(), batch_norm=True, transpose=True, scope="8_1")
        xhat = conv2d(h, [1, 1], [1, 1], 1, is_training, activation_fn=tf.nn.sigmoid, batch_norm=False, scope="mag")
    return xhat

# ============================================================================
# Fine-tuning Function
# ============================================================================

def finetune_and_evaluate(instrument_name, data_dir, test_sample_name, output_prefix):
    """Fine-tune on one instrument and generate comparison."""
    
    print("=" * 80)
    print(f"FINE-TUNING ON {instrument_name.upper()} DATA")
    print("=" * 80)
    print()
    
    # Create log directory for this instrument
    logdir = os.path.join(BASELINE_PATH, f"logs/{instrument_name}_only_finetune")
    os.makedirs(logdir, exist_ok=True)
    
    # Load training data
    print(f"Loading {instrument_name} training data...")
    wav_files = sorted(glob.glob(os.path.join(data_dir, "*.wav")))
    test_path = None
    for f in wav_files:
        if os.path.basename(f) == test_sample_name:
            test_path = f
            break
    
    if test_path is None:
        print(f"ERROR: Test sample '{test_sample_name}' not found!")
        return
    
    print(f"✓ Found {len(wav_files)} {instrument_name} samples")
    print(f"✓ Test sample: {test_sample_name}")
    
    # Load test audio
    test_audio = load_wav_file(test_path)
    if test_audio is None:
        return
    
    test_spec = prepare_spectrogram(test_audio)
    print(f"✓ Test spectrogram shape: {test_spec.shape}")
    print()
    
    # Build graph
    print("Building computation graph...")
    tf.reset_default_graph()
    
    spec_input = tf.placeholder(tf.float32, shape=[None, SPEC_HEIGHT, SPEC_WIDTH, 1], name='spec_input')
    pitch_input = tf.placeholder(tf.int32, shape=[None], name='pitch_input')
    is_training_ph = tf.placeholder(tf.bool, name='is_training')
    batch_size_ph = tf.placeholder(tf.int32, name='batch_size')
    learning_rate_ph = tf.placeholder(tf.float32, name='learning_rate')
    
    z = encode(spec_input, is_training_ph)
    spec_output = decode(z, pitch_input, batch_size_ph, is_training_ph)
    
    loss = tf.reduce_mean(tf.square(spec_input - spec_output))
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, beta1=0.5)
    train_op = optimizer.minimize(loss)
    
    print(f"✓ Encoder output: {z.shape}")
    print(f"✓ Decoder output: {spec_output.shape}")
    
    # Session and checkpoint loading
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # Load checkpoint
    reader = tf.train.NewCheckpointReader(BASELINE_CKPT)
    ckpt_vars = reader.get_variable_to_shape_map()
    graph_vars = {v.name.split(':')[0]: v for v in tf.global_variables()}
    
    vars_to_restore = {}
    for var_name, var in graph_vars.items():
        if 'Adam' in var_name or 'beta1_power' in var_name or 'beta2_power' in var_name:
            continue
        if var_name in ckpt_vars and ckpt_vars[var_name] == var.shape.as_list():
            vars_to_restore[var_name] = var
    
    if vars_to_restore:
        saver = tf.train.Saver(vars_to_restore)
        saver.restore(sess, BASELINE_CKPT)
        print(f"✓ Restored {len(vars_to_restore)} variables from checkpoint")
    print()
    
    # Baseline inference
    print("Running baseline inference...")
    test_input = test_spec[:, :, np.newaxis][np.newaxis, :, :, :]
    test_pitch = np.array([60], dtype=np.int32)
    
    baseline_out = sess.run(spec_output, feed_dict={
        spec_input: test_input, pitch_input: test_pitch,
        batch_size_ph: 1, is_training_ph: False
    })[0, :, :, 0]
    
    baseline_recon = ispecgram(baseline_out.T)[:SAMPLE_LENGTH]
    if len(baseline_recon) < SAMPLE_LENGTH:
        baseline_recon = np.pad(baseline_recon, (0, SAMPLE_LENGTH - len(baseline_recon)))
    baseline_recon = baseline_recon / (np.max(np.abs(baseline_recon)) + 1e-8)
    
    baseline_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_baseline.wav")
    sf.write(baseline_path, baseline_recon, SAMPLE_RATE)
    print(f"✓ Baseline saved: {baseline_path}")
    print()
    
    # Fine-tuning
    print(f"Fine-tuning on {instrument_name} data ({NUM_EPOCHS} epochs)...")
    losses = []
    
    for epoch in range(NUM_EPOCHS):
        epoch_losses = []
        for wav_file in wav_files:
            audio = load_wav_file(wav_file)
            if audio is None:
                continue
            spec = prepare_spectrogram(audio)
            spec_in = spec[:, :, np.newaxis][np.newaxis, :, :, :]
            
            _, loss_val = sess.run([train_op, loss], feed_dict={
                spec_input: spec_in, pitch_input: np.array([60], dtype=np.int32),
                batch_size_ph: 1, is_training_ph: True, learning_rate_ph: LEARNING_RATE
            })
            losses.append(loss_val)
            epoch_losses.append(loss_val)
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, Avg Loss: {avg_loss:.6f}")
    
    # Save fine-tuned model
    full_saver = tf.train.Saver()
    ckpt_path = os.path.join(logdir, "model.ckpt")
    full_saver.save(sess, ckpt_path)
    print(f"✓ Fine-tuned model saved: {ckpt_path}")
    print()
    
    # Fine-tuned inference
    print("Running fine-tuned inference...")
    finetune_out = sess.run(spec_output, feed_dict={
        spec_input: test_input, pitch_input: test_pitch,
        batch_size_ph: 1, is_training_ph: False
    })[0, :, :, 0]
    
    finetune_recon = ispecgram(finetune_out.T)[:SAMPLE_LENGTH]
    if len(finetune_recon) < SAMPLE_LENGTH:
        finetune_recon = np.pad(finetune_recon, (0, SAMPLE_LENGTH - len(finetune_recon)))
    finetune_recon = finetune_recon / (np.max(np.abs(finetune_recon)) + 1e-8)
    
    finetune_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_finetuned.wav")
    sf.write(finetune_path, finetune_recon, SAMPLE_RATE)
    print(f"✓ Fine-tuned saved: {finetune_path}")
    
    sess.close()
    
    # Generate spectrogram comparison
    print("Generating spectrogram comparison...")
    f, t, orig_spec = signal.spectrogram(test_audio, fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT-HOP_LENGTH)
    _, _, base_spec = signal.spectrogram(baseline_recon, fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT-HOP_LENGTH)
    _, _, fine_spec = signal.spectrogram(finetune_recon, fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT-HOP_LENGTH)
    
    orig_db = 10 * np.log10(np.abs(orig_spec) + 1e-10)
    base_db = 10 * np.log10(np.abs(base_spec) + 1e-10)
    fine_db = 10 * np.log10(np.abs(fine_spec) + 1e-10)
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    
    for idx, (spec, title) in enumerate([
        (orig_db, f'Original {instrument_name.capitalize()} Audio'),
        (base_db, f'Baseline Model (Pre-trained Checkpoint)'),
        (fine_db, f'Fine-Tuned Model ({instrument_name.capitalize()}-Trained)')
    ]):
        im = axes[idx].imshow(spec, aspect='auto', origin='lower', cmap='viridis', vmin=-80, vmax=0,
                              extent=[0, test_audio.shape[0]/SAMPLE_RATE, f[0], f[-1]])
        axes[idx].set_title(title, fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('Frequency (Hz)')
        axes[idx].set_xlabel('Time (s)')
        axes[idx].set_yscale('log')
        axes[idx].set_ylim([f[1], f[-1]])
        cbar = plt.colorbar(im, ax=axes[idx])
        cbar.set_label('Magnitude (dB)')
    
    plt.tight_layout()
    spec_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_spectrogram.png")
    plt.savefig(spec_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Spectrogram saved: {spec_path}")
    
    # Compute metrics
    baseline_metrics = compute_metrics(test_audio, baseline_recon)
    finetune_metrics = compute_metrics(test_audio, finetune_recon)
    
    improvement_mse = ((baseline_metrics['mse'] - finetune_metrics['mse']) / (baseline_metrics['mse'] + 1e-8)) * 100
    improvement_spec = ((baseline_metrics['spectral_dist'] - finetune_metrics['spectral_dist']) / 
                        (baseline_metrics['spectral_dist'] + 1e-8)) * 100
    
    print(f"\n=== {instrument_name.upper()} METRICS ===")
    print(f"Baseline MSE: {baseline_metrics['mse']:.8f}, Spectral: {baseline_metrics['spectral_dist']:.6f}")
    print(f"Fine-tuned MSE: {finetune_metrics['mse']:.8f}, Spectral: {finetune_metrics['spectral_dist']:.6f}")
    print(f"Improvement: MSE {improvement_mse:+.2f}%, Spectral {improvement_spec:+.2f}%")
    
    # Save metrics
    metrics = {
        'instrument': instrument_name,
        'test_sample': test_sample_name,
        'baseline': baseline_metrics,
        'finetuned': finetune_metrics,
        'improvement_mse_percent': improvement_mse,
        'improvement_spectral_percent': improvement_spec,
        'epochs': NUM_EPOCHS,
        'training_samples': len(wav_files),
        'final_loss': float(losses[-1]) if losses else 0,
        'checkpoint_path': ckpt_path
    }
    
    metrics_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved: {metrics_path}")
    
    # Save loss history
    loss_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_loss.json")
    with open(loss_path, 'w') as f:
        json.dump({'losses': [float(l) for l in losses]}, f)
    
    print()
    return metrics

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SEPARATE FINE-TUNING: GUITAR-ONLY AND BASS-ONLY")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Fine-tune on guitar only
    guitar_metrics = finetune_and_evaluate(
        instrument_name="guitar",
        data_dir=GUITAR_DIR,
        test_sample_name=TEST_GUITAR,
        output_prefix="guitar"
    )
    
    # Fine-tune on bass only
    bass_metrics = finetune_and_evaluate(
        instrument_name="bass",
        data_dir=BASS_DIR,
        test_sample_name=TEST_BASS,
        output_prefix="bass"
    )
    
    print("=" * 80)
    print("✓ ALL FINE-TUNING COMPLETE")
    print("=" * 80)
    print()
    print("Output files in pipeline_results3/:")
    print("  Guitar:")
    print("    • guitar_baseline.wav")
    print("    • guitar_finetuned.wav")
    print("    • guitar_spectrogram.png")
    print("    • guitar_metrics.json")
    print("  Bass:")
    print("    • bass_baseline.wav")
    print("    • bass_finetuned.wav")
    print("    • bass_spectrogram.png")
    print("    • bass_metrics.json")
    print()
    print("Fine-tuned models saved at:")
    print(f"  • {BASELINE_PATH}/logs/guitar_only_finetune/model.ckpt")
    print(f"  • {BASELINE_PATH}/logs/bass_only_finetune/model.ckpt")
