# fine_tuned.py
import os
import glob
import numpy as np
import tensorflow.compat.v1 as tf
import tf_slim as slim
import librosa

from magenta.models.nsynth import utils
from magenta.models.nsynth.baseline.models.ae_configs import nfft_1024
from magenta.models.nsynth.baseline.models import ae

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.INFO)

# ----------------------------
# USER CONFIGURATION
# ----------------------------
DATA_DIR = "data/wavs"
LOGDIR = "logs/baseline_finetune"
PRETRAINED_CKPT = "baseline-ckpt/model.ckpt-351648"
BATCH_SIZE = 1
SAMPLE_RATE = 16000
SAMPLE_LENGTH = 64000
LEARNING_RATE = 1e-5
MAX_STEPS = 10

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------

def load_wav_file(path, target_length=SAMPLE_LENGTH, sr=SAMPLE_RATE):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    # Center-crop or pad to exactly target_length
    if len(audio) < target_length:
        pad_total = target_length - len(audio)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        audio = np.pad(audio, (pad_left, pad_right), mode='constant')
    elif len(audio) > target_length:
        start = (len(audio) - target_length) // 2
        audio = audio[start:start+target_length]
    return audio.astype(np.float32)

def create_batches(file_list, batch_size=BATCH_SIZE):
    batch = []
    for file in file_list:
        audio = load_wav_file(file)
        batch.append(audio)
        if len(batch) == batch_size:
            yield np.stack(batch, axis=0)
            batch = []
    if batch:
        yield np.stack(batch, axis=0)

def tf_specgram_fixed(audio_batch, hparams):
    spec = utils.tf_specgram(
        audio_batch,
        n_fft=hparams.n_fft,
        hop_length=hparams.hop_length,
        mask=True,
        log_mag=hparams.log_mag,
        re_im=hparams.re_im,
        dphase=hparams.dphase,
        mag_only=hparams.mag_only
    )

    # tf.py_func produces unknown shape; fix rank first
    spec = tf.ensure_shape(spec, [BATCH_SIZE, None, None, None, None])

    # Remove the extra dimension -> [B, F, T, 1]
    spec = tf.squeeze(spec, axis=-1)

    # Crop freq bins 513 -> 512
    spec = spec[:, :512, :, :]

    # Force time dimension to 256
    time_dim = tf.shape(spec)[2]

    spec = tf.cond(
        time_dim < 256,
        lambda: tf.pad(spec, [[0,0],[0,0],[0,256-time_dim],[0,0]]),
        lambda: spec[:, :, :256, :]
    )

    spec.set_shape([BATCH_SIZE, 512, 256, 1])
    return spec



# ----------------------------
# MAIN TRAINING SCRIPT
# ----------------------------
def main():
    # Load WAV files
    wav_files = glob.glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True)
    if not wav_files:
        raise ValueError(f"No WAV files found in {DATA_DIR} or its subfolders")

    # ----------------------------
    # HParams
    # ----------------------------
    class HParams:
        def __init__(self):
            # Copy all hyperparameters from config_hparams
            for k, v in nfft_1024.config_hparams.items():
                setattr(self, k, v)

            self.batch_size = BATCH_SIZE
            self.learning_rate = LEARNING_RATE
            self.max_steps = MAX_STEPS
            self.sample_rate = SAMPLE_RATE
            self.raw_audio = False
            self.hop_length = 256
            self.log_mag = True
            self.adam_beta=0.5
            self.dphase = False
            self.mag_only = True
            self.fw_loss_coeff = 1.0
            self.fw_loss_cutoff = 1000
            self.cost_phase_mask = False
            self.phase_loss_coeff = 1.0
            self.re_im = False

    hparams = HParams()

    # ----------------------------
    # Placeholders
    # ----------------------------
    audio_ph = tf.placeholder(tf.float32, [BATCH_SIZE, SAMPLE_LENGTH], name="audio_ph")
    pitch_ph = tf.placeholder(tf.int32, [BATCH_SIZE], name="pitch_ph")

    # Convert audio to spectrogram
    spec = tf_specgram_fixed(audio_ph, hparams)
    batch_input = {"spectrogram": spec, "pitch": pitch_ph}

    # ----------------------------
    # Build training op
    # ----------------------------
    train_op = ae.train_op(batch_input, hparams, config_name="nfft_1024")

    # ----------------------------
    # TensorFlow session
    # ----------------------------
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver()
        if not os.path.exists(LOGDIR):
            os.makedirs(LOGDIR)

        # Restore pretrained checkpoint
        if os.path.exists(PRETRAINED_CKPT + ".index"):
            saver.restore(sess, PRETRAINED_CKPT)
            tf.logging.info(f"Loaded pretrained checkpoint: {PRETRAINED_CKPT}")
        else:
            tf.logging.warning(f"Pretrained checkpoint not found: {PRETRAINED_CKPT}, training from scratch")

        step = 0
        while step < MAX_STEPS:
            for batch_audio in create_batches(wav_files, BATCH_SIZE):
                batch_pitch = np.full((batch_audio.shape[0],), 60, dtype=np.int32)
                _, step_val = sess.run(
                    [train_op, tf.train.get_global_step()],
                    feed_dict={audio_ph: batch_audio, pitch_ph: batch_pitch}
                )
                tf.logging.info(f"Step {step_val} completed")
                step = step_val
                if step >= MAX_STEPS:
                    break

        # Save final checkpoint
        saver.save(sess, os.path.join(LOGDIR, "model.ckpt"))
        tf.logging.info(f"Training completed. Model saved to {LOGDIR}")

if __name__ == "__main__":
    main()
