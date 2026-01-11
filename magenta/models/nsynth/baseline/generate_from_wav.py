import os
import numpy as np
import tensorflow.compat.v1 as tf
import librosa
import soundfile as sf
import tf_slim as slim
from magenta.models.nsynth import utils
from magenta.models.nsynth.baseline.models import ae
from magenta.models.nsynth.baseline.models.ae_configs import nfft_1024

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.INFO)

# --- CONFIGURATION ---
INPUT_WAV = "input.wav" 
OUTPUT_WAV = "baseline_reconstruction.wav"
CHECKPOINT_DIR = "baseline-ckpt"
CHECKPOINT_FILENAME = "model.ckpt-351648"

# Model Parameters (From nfft_1024.py and ae.py defaults)
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
SAMPLE_LENGTH = 64000
CONFIG_NAME = "nfft_1024" 
GRIFFIN_LIM_ITERS = 1000

# ==============================================================================
# Helper Functions
# ==============================================================================

def audio_pad_and_center(audio, target_length):
    """
    Pads or truncates audio to exactly target_length.
    """
    current_length = len(audio)
    
    if current_length == target_length:
        return audio
        
    elif current_length < target_length:
        # Pad with zeros, centering the original audio
        padding_needed = target_length - current_length
        pad_left = padding_needed // 2
        pad_right = padding_needed - pad_left
        return np.pad(audio, (pad_left, pad_right), mode='constant')
        
    else:
        # Truncate audio, centering the window
        start_index = (current_length - target_length) // 2
        end_index = start_index + target_length
        return audio[start_index:end_index]

def specgram_np(audio, n_fft, hop_length, mask, log_mag, re_im, dphase, mag_only):
    """Spectrogram using librosa (NumPy implementation)."""
    if not hop_length:
        hop_length = int(n_fft / 2.)
    fft_config = dict(
        n_fft=n_fft, win_length=n_fft, hop_length=hop_length, center=True)
    spec = librosa.stft(audio, **fft_config)
    
    # We only care about the mag_only path used by the Baseline model
    mag, _ = librosa.core.magphase(spec)
    
    # Magnitudes, scaled 0-1 (from the original utils.specgram logic)
    if log_mag:
        mag = (librosa.power_to_db(
            mag**2, amin=1e-13, top_db=120., ref=np.max) / 120.) + 1
    else:
        mag /= mag.max()

    # The Baseline model uses mag_only=True, so only the magnitude is returned.
    # The expected shape is [Freq, Time, 1]
    spec_real = mag.astype(np.float32)[:, :, np.newaxis]
    return spec_real


def batch_specgram(audio, n_fft, hop_length, mask, log_mag, re_im, dphase, mag_only):
    """Computes specgram in a batch (Batch wrapper for specgram_np)."""
    assert len(audio.shape) == 2
    batch_size = audio.shape[0]
    res = []
    for b in range(batch_size):
        # We must replicate the exact HParams used for training the baseline
        # which were: n_fft=1024, hop_length=256, mask=True, log_mag=True, mag_only=True
        # We pass the HParams from the configuration block.
        res.append(
            specgram_np(audio[b], n_fft, hop_length, mask, log_mag, re_im, dphase,
                        mag_only))
    
    # The output shape is [Batch, Freq, Time, 1]
    return np.array(res)


def tf_specgram(audio, n_fft, hop_length, mask, log_mag, re_im, dphase, mag_only):
    """Specgram tensorflow op (uses pyfunc)."""
    # The parameters are passed to the py_func, which calls batch_specgram
    return tf.py_func(batch_specgram, [
        audio, n_fft, hop_length, mask, log_mag, re_im, dphase, mag_only
    ], tf.float32, name="tf_specgram")

# ==============================================================================
# Main Generation Function
# ==============================================================================

def run_baseline_reconstruction():
    print(f"Loading audio: {INPUT_WAV}...")
    audio, _ = librosa.load(INPUT_WAV, sr=SAMPLE_RATE)
    
    # Pad audio to the model's expected length
    audio = audio_pad_and_center(audio, SAMPLE_LENGTH)
    audio_batch = np.expand_dims(audio, 0) # [1, SAMPLE_LENGTH]

    class HParams:
        def __init__(self):
            self.n_fft = N_FFT
            self.hop_length = HOP_LENGTH
            self.sample_rate = SAMPLE_RATE
            self.log_mag = True 
            self.num_latent = nfft_1024.config_hparams.get('num_latent', 1984)
            self.mag_only = nfft_1024.config_hparams.get('mag_only', True)
            self.dphase = False 
            self.raw_audio = False 
            
    hparams = HParams()

    checkpoint_prefix = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)

    print("Building Baseline Graph...")
    with tf.Graph().as_default():
        
        # Define placeholders for audio and a dummy pitch tensor
        wav_placeholder = tf.placeholder(
            tf.float32, shape=[1, SAMPLE_LENGTH], name="wav_placeholder")
        pitch_placeholder = tf.placeholder(tf.int64, shape=[1], name="pitch_placeholder")
        
        # Convert raw audio to spectrogram using the tf_specgram function
        # The configuration parameters match the training setup (nfft_1024.py)
        specgram_input = tf_specgram(
            wav_placeholder, 
            n_fft=hparams.n_fft, 
            hop_length=hparams.hop_length, 
            mask=True,
            log_mag=hparams.log_mag, 
            re_im=False,
            dphase=False, # dphase=False because mag_only=True in nfft_1024.py config
            mag_only=hparams.mag_only
        )

        specgram_input = specgram_input[:, :512, :, :]

        # Manually set the expected input shape of the spectrogram
        expected_time_frames = SAMPLE_LENGTH // HOP_LENGTH # 250
        
        # The model's reader reduces the frequency bins by 1 (513 -> 512).
        # We must use 512 for the frequency dimension (Height).
        expected_freq_bins = (N_FFT // 2 + 1) - 1 # 513 -> 512
        
        specgram_input.set_shape([
            1, 
            expected_freq_bins, 
            expected_time_frames, 
            1
        ])
        
        # The batch dictionary for the encoder/decoder
        batch = {"spectrogram": specgram_input, "pitch": pitch_placeholder}

        # Encode: spectrogram -> latent vector z
        z = nfft_1024.encode(batch["spectrogram"], hparams, is_training=False)
            
        # Decode: latent vector z -> reconstructed spectrogram (xhat)
        xhat = nfft_1024.decode(z, batch, hparams, is_training=False)
            
        # The output shape is [Batch, Freq, Time, 1] - remove batch and channel dims
        reconstructed_spectrogram = tf.squeeze(xhat) # [Freq, Time]

        # Create a saver to load the checkpoint
        saver = tf.train.Saver()

        print("Starting Session and restoring weights...")
        with tf.Session() as sess:

            # Initialize all variables first (required for global and local variables)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # Restore the model using the discovered prefix path
            saver.restore(sess, checkpoint_prefix)
            
            pitch_dummy = np.array([60], dtype=np.int64) 
            
            print("Running inference (Autoencoding)...")
            
            # The output spectrogram is computed here
            spec_out = sess.run(
                reconstructed_spectrogram, 
                feed_dict={
                    wav_placeholder: audio_batch,
                    pitch_placeholder: pitch_dummy
                })
            
            # Add extra row of zeros
            spec_out = np.pad(spec_out, ((0, 1), (0, 0)), mode='constant')
            
            # spec_out is [Freq, Time]
            
            print("Reconstructing audio using Inverse Spectrogram and Griffin-Lim...")
            
            # We need to call batch_ispecgram (the numpy function used internally 
            # by tf_ispecgram) to convert the Spectrogram back to audio.
            # The input spectogram needs the batch dimension added back: [1, Freq, Time, 1]
            spec_out_batch = np.expand_dims(np.expand_dims(spec_out, 0), -1)

            final_audio = utils.batch_ispecgram(
                spec_out_batch,
                n_fft=hparams.n_fft,
                hop_length=hparams.hop_length,
                log_mag=hparams.log_mag,
                mag_only=hparams.mag_only,
                num_iters=GRIFFIN_LIM_ITERS
            )
            
            final_audio = np.squeeze(final_audio)

            output_dir = os.path.dirname(OUTPUT_WAV)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            print(f"Saving output to {OUTPUT_WAV}...")
            sf.write(OUTPUT_WAV, final_audio, SAMPLE_RATE)
            print("Done!")
            

if __name__ == "__main__":
    if not os.path.exists(INPUT_WAV):
        print(f"Error: Could not find {INPUT_WAV}")
    else:
        run_baseline_reconstruction()