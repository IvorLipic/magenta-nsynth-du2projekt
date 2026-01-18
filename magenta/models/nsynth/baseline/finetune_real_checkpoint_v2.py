#!/usr/bin/env python
"""
Real NSynth Fine-Tuning with Actual Checkpoint - Proper Architecture
Builds the exact same architecture as the checkpoint to enable proper weight loading.
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

print("=" * 80)
print("NSYNTH FINE-TUNING WITH REAL CHECKPOINT (Proper Architecture)")
print("=" * 80)
print()

# ============================================================================
# Configuration
# ============================================================================

WORKSPACE = "/workspaces/magenta-nsynth-du2projekt"
BASELINE_PATH = os.path.join(WORKSPACE, "magenta/models/nsynth/baseline")
GUITAR_DIR = os.path.join(BASELINE_PATH, "data/wavs/guitars")
BASS_DIR = os.path.join(BASELINE_PATH, "data/wavs/bass")
OUTPUT_DIR = os.path.join(WORKSPACE, "pipeline_results2")
BASELINE_CKPT = os.path.join(BASELINE_PATH, "baseline-ckpt/model.ckpt-351648")
FINETUNE_LOGDIR = os.path.join(BASELINE_PATH, "logs/guitar_bass_finetune")

SAMPLE_RATE = 16000
SAMPLE_LENGTH = 64000
N_FFT = 1024
HOP_LENGTH = 256
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
NUM_EPOCHS = 20  # 20 epochs for guitar + bass fine-tuning
NUM_LATENT = 1984  # From checkpoint
TEST_GUITAR = "ukelele - nice bro.wav"
TEST_BASS = "bass - guitarbass.wav"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FINETUNE_LOGDIR, exist_ok=True)

print(f"Checkpoint: {BASELINE_CKPT}")
print(f"Guitar Data: {GUITAR_DIR}")
print(f"Bass Data: {BASS_DIR}")
print()

# ============================================================================
# Helper Functions (from nsynth/utils.py)
# ============================================================================

def leaky_relu(leak=0.1):
    """Leaky ReLU activation function."""
    return lambda x: tf.maximum(x, leak * x)


def slim_batchnorm_arg_scope(is_training, activation_fn=None):
    """Create a scope for applying BatchNorm in slim."""
    batch_norm_params = {
        "is_training": is_training,
        "decay": 0.999,
        "epsilon": 0.001,
        "variables_collections": {
            "beta": None,
            "gamma": None,
            "moving_mean": "moving_vars",
            "moving_variance": "moving_vars",
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
    """2D-Conv with optional batch_norm."""
    conv_fn = slim.conv2d_transpose if transpose else slim.conv2d
    normalizer_fn = slim.batch_norm if batch_norm else None
    
    with tf.variable_scope(scope + "_Layer"):
        with slim.arg_scope(slim_batchnorm_arg_scope(is_training, activation_fn=None)):
            x = conv_fn(
                inputs=x,
                stride=stride,
                kernel_size=kernel_size,
                num_outputs=channels,
                normalizer_fn=normalizer_fn,
                biases_initializer=tf.zeros_initializer(),
                scope=scope)
            if activation_fn:
                x = activation_fn(x)
    return x


def pitch_embeddings(pitch, batch_size, n_pitches=128, dim_embedding=128):
    """Get embedding of each pitch note."""
    with tf.variable_scope("PitchEmbedding"):
        w = tf.get_variable(
            name="embedding_weights",
            shape=[n_pitches, dim_embedding],
            initializer=tf.random_normal_initializer())
        one_hot_pitch = tf.reshape(pitch, [batch_size])
        one_hot_pitch = tf.one_hot(one_hot_pitch, depth=n_pitches)
        embedding = tf.matmul(one_hot_pitch, w)
        embedding = tf.reshape(embedding, [batch_size, 1, 1, dim_embedding])
    return embedding


def specgram(audio, n_fft=1024, hop_length=256):
    """Compute magnitude spectrogram using scipy."""
    # Use scipy STFT
    f, t, Zxx = signal.stft(audio, fs=SAMPLE_RATE, nperseg=n_fft, noverlap=n_fft-hop_length)
    mag = np.abs(Zxx)
    # Normalize to [0, 1]
    mag = mag / (np.max(mag) + 1e-8)
    return mag.T  # [time, freq]


def ispecgram(mag, n_fft=1024, hop_length=256, n_iter=50):
    """Reconstruct audio from magnitude spectrogram using Griffin-Lim."""
    # mag shape: [time, freq]
    # Need to transpose to [freq, time] for istft
    mag = mag.T  # [freq, time]
    
    # Pad freq to n_fft//2 + 1 = 513 if needed
    target_freq = n_fft // 2 + 1
    if mag.shape[0] < target_freq:
        mag = np.pad(mag, ((0, target_freq - mag.shape[0]), (0, 0)), mode='constant')
    elif mag.shape[0] > target_freq:
        mag = mag[:target_freq, :]
    
    # Griffin-Lim algorithm
    # Initialize with random phase
    angle = np.exp(2j * np.pi * np.random.rand(*mag.shape))
    S = mag * angle
    
    for _ in range(n_iter):
        # ISTFT
        _, audio = signal.istft(S, fs=SAMPLE_RATE, nperseg=n_fft, noverlap=n_fft-hop_length)
        # STFT
        _, _, S_new = signal.stft(audio, fs=SAMPLE_RATE, nperseg=n_fft, noverlap=n_fft-hop_length)
        # Keep magnitude, update phase
        angle = np.exp(1j * np.angle(S_new))
        S = mag * angle
    
    _, audio = signal.istft(S, fs=SAMPLE_RATE, nperseg=n_fft, noverlap=n_fft-hop_length)
    return audio.astype(np.float32)


# ============================================================================
# Model Architecture (matching checkpoint exactly)
# ============================================================================

def encode(x, is_training=True):
    """Encoder network matching checkpoint architecture."""
    with tf.variable_scope("encoder"):
        h = conv2d(x, [5, 5], [2, 2], 128, is_training, 
                   activation_fn=leaky_relu(), batch_norm=True, scope="0")
        h = conv2d(h, [4, 4], [2, 2], 128, is_training,
                   activation_fn=leaky_relu(), batch_norm=True, scope="1")
        h = conv2d(h, [4, 4], [2, 2], 128, is_training,
                   activation_fn=leaky_relu(), batch_norm=True, scope="2")
        h = conv2d(h, [4, 4], [2, 2], 256, is_training,
                   activation_fn=leaky_relu(), batch_norm=True, scope="3")
        h = conv2d(h, [4, 4], [2, 2], 256, is_training,
                   activation_fn=leaky_relu(), batch_norm=True, scope="4")
        h = conv2d(h, [4, 4], [2, 2], 256, is_training,
                   activation_fn=leaky_relu(), batch_norm=True, scope="5")
        h = conv2d(h, [4, 4], [2, 2], 512, is_training,
                   activation_fn=leaky_relu(), batch_norm=True, scope="6")
        h = conv2d(h, [4, 4], [2, 2], 512, is_training,
                   activation_fn=leaky_relu(), batch_norm=True, scope="7")
        h = conv2d(h, [4, 4], [2, 1], 512, is_training,
                   activation_fn=leaky_relu(), batch_norm=True, scope="7_1")
        h = conv2d(h, [1, 1], [1, 1], 1024, is_training,
                   activation_fn=leaky_relu(), batch_norm=True, scope="8")
        z = conv2d(h, [1, 1], [1, 1], NUM_LATENT, is_training,
                   activation_fn=None, batch_norm=True, scope="z")
    return z


def decode(z, pitch, batch_size, is_training=True):
    """Decoder network matching checkpoint architecture."""
    with tf.variable_scope("decoder"):
        z_pitch = pitch_embeddings(pitch, batch_size)
        z = tf.concat([z, z_pitch], 3)
        
        h = conv2d(z, [1, 1], [1, 1], 1024, is_training,
                   activation_fn=leaky_relu(), batch_norm=True, transpose=True, scope="0")
        h = conv2d(h, [4, 4], [2, 2], 512, is_training,
                   activation_fn=leaky_relu(), batch_norm=True, transpose=True, scope="1")
        h = conv2d(h, [4, 4], [2, 2], 512, is_training,
                   activation_fn=leaky_relu(), batch_norm=True, transpose=True, scope="2")
        h = conv2d(h, [4, 4], [2, 2], 256, is_training,
                   activation_fn=leaky_relu(), batch_norm=True, transpose=True, scope="3")
        h = conv2d(h, [4, 4], [2, 2], 256, is_training,
                   activation_fn=leaky_relu(), batch_norm=True, transpose=True, scope="4")
        h = conv2d(h, [4, 4], [2, 2], 256, is_training,
                   activation_fn=leaky_relu(), batch_norm=True, transpose=True, scope="5")
        h = conv2d(h, [4, 4], [2, 2], 128, is_training,
                   activation_fn=leaky_relu(), batch_norm=True, transpose=True, scope="6")
        h = conv2d(h, [4, 4], [2, 2], 128, is_training,
                   activation_fn=leaky_relu(), batch_norm=True, transpose=True, scope="7")
        h = conv2d(h, [5, 5], [2, 2], 128, is_training,
                   activation_fn=leaky_relu(), batch_norm=True, transpose=True, scope="8")
        h = conv2d(h, [5, 5], [2, 1], 128, is_training,
                   activation_fn=leaky_relu(), batch_norm=True, transpose=True, scope="8_1")
        xhat = conv2d(h, [1, 1], [1, 1], 1, is_training,
                      activation_fn=tf.nn.sigmoid, batch_norm=False, scope="mag")
    return xhat


# ============================================================================
# Load Guitar and Bass Data
# ============================================================================

print("STEP 1: Loading guitar and bass training data...")
print("-" * 80)

# Load guitar files
guitar_wav_files = sorted(glob.glob(os.path.join(GUITAR_DIR, "*.wav")))
bass_wav_files = sorted(glob.glob(os.path.join(BASS_DIR, "*.wav")))

# Find test samples
test_guitar_path = None
test_bass_path = None
training_wav_files = []

for f in guitar_wav_files:
    basename = os.path.basename(f)
    if basename == TEST_GUITAR:
        test_guitar_path = f
    training_wav_files.append(f)

for f in bass_wav_files:
    basename = os.path.basename(f)
    if basename == TEST_BASS:
        test_bass_path = f
    training_wav_files.append(f)

if test_guitar_path is None:
    print(f"ERROR: Guitar test sample '{TEST_GUITAR}' not found!")
    sys.exit(1)
if test_bass_path is None:
    print(f"ERROR: Bass test sample '{TEST_BASS}' not found!")
    sys.exit(1)

print(f"✓ Found {len(guitar_wav_files)} guitar samples")
print(f"✓ Found {len(bass_wav_files)} bass samples")
print(f"✓ Total training samples: {len(training_wav_files)}")
print(f"\nGuitar samples:")
for i, f in enumerate(guitar_wav_files[:3]):
    print(f"  {i+1}. {os.path.basename(f)}")
if len(guitar_wav_files) > 3:
    print(f"  ... and {len(guitar_wav_files) - 3} more")
print(f"\nBass samples:")
for i, f in enumerate(bass_wav_files[:3]):
    print(f"  {i+1}. {os.path.basename(f)}")
if len(bass_wav_files) > 3:
    print(f"  ... and {len(bass_wav_files) - 3} more")
print()


def load_wav_file(path, target_length=SAMPLE_LENGTH, sr=SAMPLE_RATE):
    """Load and preprocess audio."""
    try:
        audio, file_sr = sf.read(path)
        if file_sr != sr:
            num_samples = int(len(audio) * sr / file_sr)
            audio = signal.resample(audio, num_samples)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        if len(audio) < target_length:
            pad_total = target_length - len(audio)
            pad_left = pad_total // 2
            audio = np.pad(audio, (pad_left, pad_total - pad_left), mode='constant')
        elif len(audio) > target_length:
            start = (len(audio) - target_length) // 2
            audio = audio[start:start + target_length]
        return audio.astype(np.float32)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


# Load both test samples
test_guitar_audio = load_wav_file(test_guitar_path)
test_bass_audio = load_wav_file(test_bass_path)
if test_guitar_audio is None or test_bass_audio is None:
    sys.exit(1)

print(f"✓ Guitar test sample loaded: {os.path.basename(test_guitar_path)} - {test_guitar_audio.shape}")
print(f"✓ Bass test sample loaded: {os.path.basename(test_bass_path)} - {test_bass_audio.shape}")
print()

# ============================================================================
# Compute Spectrogram for Test Audio
# ============================================================================

print("STEP 2: Computing spectrograms...")
print("-" * 80)

# Model expects specific dimensions: [batch, 512, 256, 1]
# 512 = freq bins (downsampled from 513), 256 = time frames
SPEC_HEIGHT = 512  # freq dimension
SPEC_WIDTH = 256   # time dimension


def prepare_spectrogram(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Compute and pad spectrogram to model dimensions."""
    spec = specgram(audio, n_fft=n_fft, hop_length=hop_length)
    # spec shape is [time, freq]
    time_frames, freq_bins = spec.shape
    
    # Transpose to [freq, time] for the model
    spec = spec.T  # [freq, time]
    
    # Pad/crop freq to 512 (from 513)
    if freq_bins > SPEC_HEIGHT:
        spec = spec[:SPEC_HEIGHT, :]
    elif freq_bins < SPEC_HEIGHT:
        spec = np.pad(spec, ((0, SPEC_HEIGHT - freq_bins), (0, 0)), mode='constant')
    
    # Pad/crop time to 256
    if spec.shape[1] > SPEC_WIDTH:
        spec = spec[:, :SPEC_WIDTH]
    elif spec.shape[1] < SPEC_WIDTH:
        spec = np.pad(spec, ((0, 0), (0, SPEC_WIDTH - spec.shape[1])), mode='constant')
    
    return spec  # [512, 256]


test_guitar_spec = prepare_spectrogram(test_guitar_audio)
test_bass_spec = prepare_spectrogram(test_bass_audio)
print(f"✓ Guitar spectrogram shape: {test_guitar_spec.shape}")
print(f"✓ Bass spectrogram shape: {test_bass_spec.shape}")

# Shape for the model: [batch, height=freq, width=time, channels=1]
spec_height = SPEC_HEIGHT
spec_width = SPEC_WIDTH
spec_channels = 1

print(f"✓ Model input dimensions: freq={spec_height}, time={spec_width}, channels={spec_channels}")
print()

# ============================================================================
# Build Graph and Load Checkpoint
# ============================================================================

print("STEP 3: Building computation graph...")
print("-" * 80)

tf.reset_default_graph()

# Input placeholder: [batch, freq, time, channels=1]
# NSynth model expects [batch, height, width, channels]
spec_input = tf.placeholder(tf.float32, shape=[None, spec_height, spec_width, spec_channels], name='spec_input')
pitch_input = tf.placeholder(tf.int32, shape=[None], name='pitch_input')
is_training_ph = tf.placeholder(tf.bool, name='is_training')
batch_size_ph = tf.placeholder(tf.int32, name='batch_size')

# Build encoder
z = encode(spec_input, is_training_ph)
print(f"✓ Encoder built - latent shape: {z.shape}")

# Build decoder
spec_output = decode(z, pitch_input, batch_size_ph, is_training_ph)
print(f"✓ Decoder built - output shape: {spec_output.shape}")

# Loss for fine-tuning
loss = tf.reduce_mean(tf.square(spec_input - spec_output))

# Optimizer
learning_rate_ph = tf.placeholder(tf.float32, name='learning_rate')
optimizer = tf.train.AdamOptimizer(learning_rate_ph, beta1=0.5)

# Get trainable variables (excluding optimizer variables)
model_vars = [v for v in tf.trainable_variables() 
              if 'Adam' not in v.name and 'beta' not in v.name.split('/')[-1].split(':')[0]
              or 'BatchNorm/beta' in v.name]
train_op = optimizer.minimize(loss)

# Count variables
total_vars = len(tf.global_variables())
trainable_vars = len(tf.trainable_variables())
print(f"✓ Total variables: {total_vars}, Trainable: {trainable_vars}")
print()

# ============================================================================
# Load Checkpoint
# ============================================================================

print("STEP 4: Loading checkpoint...")
print("-" * 80)

# Create session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Get variables in checkpoint
reader = tf.train.NewCheckpointReader(BASELINE_CKPT)
ckpt_vars = reader.get_variable_to_shape_map()
print(f"✓ Checkpoint has {len(ckpt_vars)} variables")

# Get model variables (excluding optimizer state)
graph_vars = {v.name.split(':')[0]: v for v in tf.global_variables()}

# Find matching variables
vars_to_restore = {}
matched = 0
unmatched_ckpt = []
unmatched_graph = []

for var_name, var in graph_vars.items():
    # Skip optimizer variables
    if 'Adam' in var_name or 'beta1_power' in var_name or 'beta2_power' in var_name:
        continue
    if var_name in ckpt_vars:
        ckpt_shape = ckpt_vars[var_name]
        graph_shape = var.shape.as_list()
        if ckpt_shape == graph_shape:
            vars_to_restore[var_name] = var
            matched += 1
        else:
            print(f"  Shape mismatch: {var_name} - ckpt:{ckpt_shape} vs graph:{graph_shape}")
    else:
        unmatched_graph.append(var_name)

# Check what's in checkpoint but not in graph
for ckpt_name in ckpt_vars:
    if ckpt_name not in graph_vars and 'Adam' not in ckpt_name and 'beta' not in ckpt_name.split('/')[-1]:
        if not ckpt_name.startswith('Optimizer'):
            unmatched_ckpt.append(ckpt_name)

print(f"✓ Matched {matched} variables for restoration")

if unmatched_ckpt:
    print(f"⚠ {len(unmatched_ckpt)} checkpoint vars not in graph:")
    for v in unmatched_ckpt[:5]:
        print(f"    - {v}")
    if len(unmatched_ckpt) > 5:
        print(f"    ... and {len(unmatched_ckpt)-5} more")

if unmatched_graph:
    print(f"⚠ {len(unmatched_graph)} graph vars not in checkpoint:")
    for v in unmatched_graph[:5]:
        print(f"    - {v}")

# Restore matched variables
if vars_to_restore:
    saver = tf.train.Saver(vars_to_restore)
    saver.restore(sess, BASELINE_CKPT)
    print(f"✓ Successfully restored {len(vars_to_restore)} variables from checkpoint!")
    weights_loaded = True
else:
    print("✗ No matching variables found!")
    weights_loaded = False
print()

# ============================================================================
# Run Baseline Inference
# ============================================================================

print("STEP 5: Running baseline model inference...")
print("-" * 80)

# Prepare guitar input: [batch, freq=512, time=256, channels=1]
test_guitar_input = test_guitar_spec[:, :, np.newaxis][np.newaxis, :, :, :]
test_bass_input = test_bass_spec[:, :, np.newaxis][np.newaxis, :, :, :]
print(f"✓ Guitar input shape: {test_guitar_input.shape}")
print(f"✓ Bass input shape: {test_bass_input.shape}")

# Use middle pitch (60 = middle C)
test_pitch = np.array([60], dtype=np.int32)

# Baseline guitar reconstruction
baseline_guitar_out = sess.run(spec_output, feed_dict={
    spec_input: test_guitar_input,
    pitch_input: test_pitch,
    batch_size_ph: 1,
    is_training_ph: False
})
baseline_guitar_out = baseline_guitar_out[0, :, :, 0]

# Baseline bass reconstruction
baseline_bass_out = sess.run(spec_output, feed_dict={
    spec_input: test_bass_input,
    pitch_input: test_pitch,
    batch_size_ph: 1,
    is_training_ph: False
})
baseline_bass_out = baseline_bass_out[0, :, :, 0]

print(f"✓ Baseline guitar output shape: {baseline_guitar_out.shape}")
print(f"✓ Baseline bass output shape: {baseline_bass_out.shape}")

# Reconstruct audio from spectrograms
baseline_guitar_recon = ispecgram(baseline_guitar_out.T, n_fft=N_FFT, hop_length=HOP_LENGTH)
baseline_guitar_recon = baseline_guitar_recon[:SAMPLE_LENGTH]
if len(baseline_guitar_recon) < SAMPLE_LENGTH:
    baseline_guitar_recon = np.pad(baseline_guitar_recon, (0, SAMPLE_LENGTH - len(baseline_guitar_recon)))
baseline_guitar_recon = baseline_guitar_recon / (np.max(np.abs(baseline_guitar_recon)) + 1e-8)

baseline_bass_recon = ispecgram(baseline_bass_out.T, n_fft=N_FFT, hop_length=HOP_LENGTH)
baseline_bass_recon = baseline_bass_recon[:SAMPLE_LENGTH]
if len(baseline_bass_recon) < SAMPLE_LENGTH:
    baseline_bass_recon = np.pad(baseline_bass_recon, (0, SAMPLE_LENGTH - len(baseline_bass_recon)))
baseline_bass_recon = baseline_bass_recon / (np.max(np.abs(baseline_bass_recon)) + 1e-8)

sf.write(os.path.join(OUTPUT_DIR, "baseline_guitar_reconstruction.wav"), baseline_guitar_recon, SAMPLE_RATE)
sf.write(os.path.join(OUTPUT_DIR, "baseline_bass_reconstruction.wav"), baseline_bass_recon, SAMPLE_RATE)
print(f"✓ Baseline guitar reconstruction saved")
print(f"✓ Baseline bass reconstruction saved")
print()

# ============================================================================
# Fine-Tuning
# ============================================================================

print("STEP 6: Fine-tuning on guitar and bass data...")
print("-" * 80)

losses = []
step = 0
total_steps = len(training_wav_files) * NUM_EPOCHS

for epoch in range(NUM_EPOCHS):
    epoch_losses = []
    for wav_file in training_wav_files:
        audio = load_wav_file(wav_file)
        if audio is None:
            continue
        
        # Compute and prepare spectrogram: [freq, time] -> [batch, freq, time, 1]
        spec = prepare_spectrogram(audio)
        spec_in = spec[:, :, np.newaxis]  # [512, 256, 1]
        spec_in = spec_in[np.newaxis, :, :, :]  # [1, 512, 256, 1]
        
        # Training step
        _, loss_val = sess.run([train_op, loss], feed_dict={
            spec_input: spec_in,
            pitch_input: np.array([60], dtype=np.int32),
            batch_size_ph: 1,
            is_training_ph: True,
            learning_rate_ph: LEARNING_RATE
        })
        
        losses.append(loss_val)
        epoch_losses.append(loss_val)
        step += 1
    
    avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0
    # Print every epoch for visibility
    print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, Avg Loss: {avg_epoch_loss:.6f}")

print(f"✓ Fine-tuning complete ({step} steps)")

# Save fine-tuned checkpoint
full_saver = tf.train.Saver()
finetune_ckpt_path = os.path.join(FINETUNE_LOGDIR, "model.ckpt")
full_saver.save(sess, finetune_ckpt_path)
print(f"✓ Fine-tuned model saved: {finetune_ckpt_path}")
print()

# ============================================================================
# Fine-tuned Inference
# ============================================================================

print("STEP 7: Running fine-tuned model inference...")
print("-" * 80)

# Fine-tuned guitar reconstruction
finetune_guitar_out = sess.run(spec_output, feed_dict={
    spec_input: test_guitar_input,
    pitch_input: test_pitch,
    batch_size_ph: 1,
    is_training_ph: False
})
finetune_guitar_out = finetune_guitar_out[0, :, :, 0]

# Fine-tuned bass reconstruction
finetune_bass_out = sess.run(spec_output, feed_dict={
    spec_input: test_bass_input,
    pitch_input: test_pitch,
    batch_size_ph: 1,
    is_training_ph: False
})
finetune_bass_out = finetune_bass_out[0, :, :, 0]

print(f"✓ Fine-tuned guitar output shape: {finetune_guitar_out.shape}")
print(f"✓ Fine-tuned bass output shape: {finetune_bass_out.shape}")

# Reconstruct audio
finetune_guitar_recon = ispecgram(finetune_guitar_out.T, n_fft=N_FFT, hop_length=HOP_LENGTH)
finetune_guitar_recon = finetune_guitar_recon[:SAMPLE_LENGTH]
if len(finetune_guitar_recon) < SAMPLE_LENGTH:
    finetune_guitar_recon = np.pad(finetune_guitar_recon, (0, SAMPLE_LENGTH - len(finetune_guitar_recon)))
finetune_guitar_recon = finetune_guitar_recon / (np.max(np.abs(finetune_guitar_recon)) + 1e-8)

finetune_bass_recon = ispecgram(finetune_bass_out.T, n_fft=N_FFT, hop_length=HOP_LENGTH)
finetune_bass_recon = finetune_bass_recon[:SAMPLE_LENGTH]
if len(finetune_bass_recon) < SAMPLE_LENGTH:
    finetune_bass_recon = np.pad(finetune_bass_recon, (0, SAMPLE_LENGTH - len(finetune_bass_recon)))
finetune_bass_recon = finetune_bass_recon / (np.max(np.abs(finetune_bass_recon)) + 1e-8)

sf.write(os.path.join(OUTPUT_DIR, "finetune_guitar_reconstruction.wav"), finetune_guitar_recon, SAMPLE_RATE)
sf.write(os.path.join(OUTPUT_DIR, "finetune_bass_reconstruction.wav"), finetune_bass_recon, SAMPLE_RATE)
print(f"✓ Fine-tuned guitar reconstruction saved")
print(f"✓ Fine-tuned bass reconstruction saved")
print()

# Close session
sess.close()

# ============================================================================
# Generate Spectrograms
# ============================================================================

print("STEP 8: Generating spectrograms...")
print("-" * 80)

# Guitar spectrograms
f, t, original_guitar_spec = signal.spectrogram(test_guitar_audio, fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT-HOP_LENGTH)
_, _, baseline_guitar_spec = signal.spectrogram(baseline_guitar_recon, fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT-HOP_LENGTH)
_, _, finetune_guitar_spec = signal.spectrogram(finetune_guitar_recon, fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT-HOP_LENGTH)

# Bass spectrograms
_, _, original_bass_spec = signal.spectrogram(test_bass_audio, fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT-HOP_LENGTH)
_, _, baseline_bass_spec = signal.spectrogram(baseline_bass_recon, fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT-HOP_LENGTH)
_, _, finetune_bass_spec = signal.spectrogram(finetune_bass_recon, fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT-HOP_LENGTH)

# Convert to dB
original_guitar_db = 10 * np.log10(np.abs(original_guitar_spec) + 1e-10)
baseline_guitar_db = 10 * np.log10(np.abs(baseline_guitar_spec) + 1e-10)
finetune_guitar_db = 10 * np.log10(np.abs(finetune_guitar_spec) + 1e-10)
original_bass_db = 10 * np.log10(np.abs(original_bass_spec) + 1e-10)
baseline_bass_db = 10 * np.log10(np.abs(baseline_bass_spec) + 1e-10)
finetune_bass_db = 10 * np.log10(np.abs(finetune_bass_spec) + 1e-10)

# Create 2x3 spectrogram comparison (Guitar row, Bass row)
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

# Guitar row
for idx, (spec, title) in enumerate([
    (original_guitar_db, 'Original Guitar'),
    (baseline_guitar_db, 'Baseline Model (Guitar)'),
    (finetune_guitar_db, 'Fine-Tuned Model (Guitar)')
]):
    im = axes[0, idx].imshow(spec, aspect='auto', origin='lower', cmap='viridis', vmin=-80, vmax=0,
                              extent=[0, test_guitar_audio.shape[0]/SAMPLE_RATE, f[0], f[-1]])
    axes[0, idx].set_title(title, fontsize=12, fontweight='bold')
    axes[0, idx].set_ylabel('Frequency (Hz)')
    axes[0, idx].set_xlabel('Time (s)')
    axes[0, idx].set_yscale('log')
    axes[0, idx].set_ylim([f[1], f[-1]])
    plt.colorbar(im, ax=axes[0, idx], label='dB')

# Bass row
for idx, (spec, title) in enumerate([
    (original_bass_db, 'Original Bass'),
    (baseline_bass_db, 'Baseline Model (Bass)'),
    (finetune_bass_db, 'Fine-Tuned Model (Bass)')
]):
    im = axes[1, idx].imshow(spec, aspect='auto', origin='lower', cmap='viridis', vmin=-80, vmax=0,
                              extent=[0, test_bass_audio.shape[0]/SAMPLE_RATE, f[0], f[-1]])
    axes[1, idx].set_title(title, fontsize=12, fontweight='bold')
    axes[1, idx].set_ylabel('Frequency (Hz)')
    axes[1, idx].set_xlabel('Time (s)')
    axes[1, idx].set_yscale('log')
    axes[1, idx].set_ylim([f[1], f[-1]])
    plt.colorbar(im, ax=axes[1, idx], label='dB')

plt.tight_layout()
spec_path = os.path.join(OUTPUT_DIR, "spectrogram_guitar_bass_comparison.png")
plt.savefig(spec_path, dpi=150, bbox_inches='tight')
print(f"✓ Spectrogram saved: {spec_path}")
plt.close()

# ============================================================================
# Metrics
# ============================================================================

print("STEP 9: Computing metrics...")
print("-" * 80)

def compute_metrics(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(original - reconstructed))
    f_o, t_o, spec_o = signal.spectrogram(original, fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT-HOP_LENGTH)
    f_r, t_r, spec_r = signal.spectrogram(reconstructed, fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT-HOP_LENGTH)
    spec_o = spec_o / (np.max(np.abs(spec_o)) + 1e-8)
    spec_r = spec_r / (np.max(np.abs(spec_r)) + 1e-8)
    min_len = min(spec_o.shape[1], spec_r.shape[1])
    spec_dist = np.mean(np.sqrt(np.sum((spec_o[:, :min_len] - spec_r[:, :min_len]) ** 2, axis=0)))
    return {'mse': float(mse), 'rmse': float(rmse), 'mae': float(mae), 'spectral_dist': float(spec_dist)}

# Guitar metrics
baseline_guitar_metrics = compute_metrics(test_guitar_audio, baseline_guitar_recon)
finetune_guitar_metrics = compute_metrics(test_guitar_audio, finetune_guitar_recon)

# Bass metrics
baseline_bass_metrics = compute_metrics(test_bass_audio, baseline_bass_recon)
finetune_bass_metrics = compute_metrics(test_bass_audio, finetune_bass_recon)

print(f"\n=== GUITAR ===")
print(f"Baseline Model:")
print(f"  MSE: {baseline_guitar_metrics['mse']:.8f}")
print(f"  Spectral Distance: {baseline_guitar_metrics['spectral_dist']:.6f}")
print(f"Fine-Tuned Model:")
print(f"  MSE: {finetune_guitar_metrics['mse']:.8f}")
print(f"  Spectral Distance: {finetune_guitar_metrics['spectral_dist']:.6f}")

print(f"\n=== BASS ===")
print(f"Baseline Model:")
print(f"  MSE: {baseline_bass_metrics['mse']:.8f}")
print(f"  Spectral Distance: {baseline_bass_metrics['spectral_dist']:.6f}")
print(f"Fine-Tuned Model:")
print(f"  MSE: {finetune_bass_metrics['mse']:.8f}")
print(f"  Spectral Distance: {finetune_bass_metrics['spectral_dist']:.6f}")

guitar_improvement_mse = ((baseline_guitar_metrics['mse'] - finetune_guitar_metrics['mse']) / (baseline_guitar_metrics['mse'] + 1e-8)) * 100
guitar_improvement_spectral = ((baseline_guitar_metrics['spectral_dist'] - finetune_guitar_metrics['spectral_dist']) / 
                               (baseline_guitar_metrics['spectral_dist'] + 1e-8)) * 100
bass_improvement_mse = ((baseline_bass_metrics['mse'] - finetune_bass_metrics['mse']) / (baseline_bass_metrics['mse'] + 1e-8)) * 100
bass_improvement_spectral = ((baseline_bass_metrics['spectral_dist'] - finetune_bass_metrics['spectral_dist']) / 
                             (baseline_bass_metrics['spectral_dist'] + 1e-8)) * 100

print(f"\n✓ Guitar Improvement: MSE {guitar_improvement_mse:+.2f}%, Spectral {guitar_improvement_spectral:+.2f}%")
print(f"✓ Bass Improvement: MSE {bass_improvement_mse:+.2f}%, Spectral {bass_improvement_spectral:+.2f}%")
print()

# Save metrics
metrics = {
    'guitar': {
        'baseline': baseline_guitar_metrics,
        'finetune': finetune_guitar_metrics,
        'improvement_mse_percent': guitar_improvement_mse,
        'improvement_spectral_percent': guitar_improvement_spectral
    },
    'bass': {
        'baseline': baseline_bass_metrics,
        'finetune': finetune_bass_metrics,
        'improvement_mse_percent': bass_improvement_mse,
        'improvement_spectral_percent': bass_improvement_spectral
    },
    'config': {
        'checkpoint': BASELINE_CKPT,
        'guitar_samples': len(guitar_wav_files),
        'bass_samples': len(bass_wav_files),
        'total_samples': len(training_wav_files),
        'epochs': NUM_EPOCHS,
        'weights_loaded': weights_loaded,
        'num_vars_restored': len(vars_to_restore) if weights_loaded else 0
    },
    'training_loss': {
        'final': float(losses[-1]) if losses else 0,
        'min': float(np.min(losses)) if losses else 0,
        'avg': float(np.mean(losses)) if losses else 0,
    }
}

with open(os.path.join(OUTPUT_DIR, "metrics_guitar_bass.json"), 'w') as f:
    json.dump(metrics, f, indent=2)

# Save loss history
with open(os.path.join(OUTPUT_DIR, "training_loss_guitar_bass.json"), 'w') as f:
    json.dump({'losses': [float(l) for l in losses]}, f)

print("=" * 80)
print("✓ GUITAR + BASS FINE-TUNING COMPLETE")
print("=" * 80)
print()
print("Output files:")
print(f"  • baseline_guitar_reconstruction.wav")
print(f"  • baseline_bass_reconstruction.wav")
print(f"  • finetune_guitar_reconstruction.wav")
print(f"  • finetune_bass_reconstruction.wav")
print(f"  • spectrogram_guitar_bass_comparison.png")
print(f"  • metrics_guitar_bass.json")
print(f"  • training_loss_guitar_bass.json")
print()
