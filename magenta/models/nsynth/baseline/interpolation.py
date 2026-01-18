import librosa
import numpy as np
from magenta.models.nsynth.baseline.models.ae_configs import nfft_1024
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import generate_from_wav
import os

# Disable eager execution for TF 1.x compatibility
tf.disable_eager_execution()

def generate_interpolations(model_params, spec1, spec2, num_steps):
    """Generate interpolations between two spectrograms using the given model.

    Args:
        model_params: Hyperparameters for the model (dict).
        spec1: The first input spectrogram tensor.
        spec2: The second input spectrogram tensor.
        num_steps: The number of interpolation steps to generate.

    Returns:
        A list of interpolated spectrogram tensors.
    """
    # Encode the input spectrograms to latent space
    z1 = nfft_1024.encode(spec1, model_params, is_training=False, reuse=False)
    z2 = nfft_1024.encode(spec2, model_params, is_training=False, reuse=True)

    # Use spec1 as the spectrogram input (already sliced)
    pitch_placeholder = tf.placeholder(tf.int64, shape=[1], name="pitch_placeholder")
    batch = {"spectrogram": spec1, "pitch": pitch_placeholder}

    interpolations = []
    for alpha in np.linspace(0, 1, num_steps):
        # Linear interpolation in latent space
        z_interp = (1 - alpha) * z1 + alpha * z2
        # Decode back to spectrogram space (reuse variables)
        spec_interp = nfft_1024.decode(z_interp, batch, model_params, is_training=False, reuse=tf.AUTO_REUSE)
        interpolations.append(spec_interp)

    return interpolations

if __name__ == "__main__":
    # Example usage
    class HParams:
        def __init__(self):
            self.n_fft = generate_from_wav.N_FFT
            self.hop_length = generate_from_wav.HOP_LENGTH
            self.sample_rate = generate_from_wav.SAMPLE_RATE
            self.log_mag = True 
            self.num_latent = nfft_1024.config_hparams.get('num_latent', 1984)
            self.mag_only = nfft_1024.config_hparams.get('mag_only', True)
            self.dphase = False 
            self.raw_audio = False 
            
    model_params = HParams()
    num_steps = 3

    audio1, _ = librosa.load('./data/wavs/bell - business.wav', sr=16000)
    audio2, _ = librosa.load('./data/wavs/guitar - hard.wav', sr=16000)

    audio1 = generate_from_wav.audio_pad_and_center(audio1, 16384)
    audio2 = generate_from_wav.audio_pad_and_center(audio2, 16384)

    # Expand to batch dimension [1, audio_length]
    audio1_batch = np.expand_dims(audio1, 0)
    audio2_batch = np.expand_dims(audio2, 0)

    with tf.Graph().as_default():
        # Compute spectrograms
        spec1 = generate_from_wav.tf_specgram(
            audio1_batch, n_fft=1024, hop_length=256, mask=True,
            log_mag=True, re_im=False, dphase=False, mag_only=True)
        spec2 = generate_from_wav.tf_specgram(
            audio2_batch, n_fft=1024, hop_length=256, mask=True,
            log_mag=True, re_im=False, dphase=False, mag_only=True)
        
        # Set shape explicitly
        spec1.set_shape([1, 513, None, 1])
        spec2.set_shape([1, 513, None, 1])
        
        # Slice to 512 frequency bins
        spec1 = spec1[:, :512, :, :]
        spec2 = spec2[:, :512, :, :]
        
        # Generate interpolations
        interpolated_specs = generate_interpolations(model_params, spec1, spec2, num_steps)
        
        # Run the graph in a session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Feed pitch
            feed_dict = {tf.get_default_graph().get_tensor_by_name("pitch_placeholder:0"): [60]}
            results = sess.run(interpolated_specs, feed_dict=feed_dict)
            for i, spec in enumerate(results):
                print(f"Interpolation {i}: shape {spec.shape}")

            # Visualize original spectrograms
            spec1_result = sess.run(spec1, feed_dict=feed_dict)
            spec2_result = sess.run(spec2, feed_dict=feed_dict)

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(spec1_result[0, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(label='Log Magnitude')
            plt.xlabel('Time Frames')
            plt.ylabel('Frequency Bins')
            plt.title('Spectrogram 1 (Bell)')
            plt.tight_layout()

            plt.subplot(1, 2, 2)
            plt.imshow(spec2_result[0, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(label='Log Magnitude')
            plt.xlabel('Time Frames')
            plt.ylabel('Frequency Bins')
            plt.title('Spectrogram 2 (Guitar)')
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(12, 3 * len(results)))
            for i, spec in enumerate(results):
                plt.subplot(len(results), 1, i + 1)
                plt.imshow(spec[0, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
                plt.colorbar(label='Log Magnitude')
                plt.xlabel('Time Frames')
                plt.ylabel('Frequency Bins')
                plt.title(f'Interpolation {i}')
            plt.tight_layout()
            plt.show()