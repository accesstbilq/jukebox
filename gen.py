# import os
# import torch as t
# import librosa
# from jukebox.make_models import make_vqvae, make_prior, MODELS
# from jukebox.hparams import Hyperparams, setup_hparams
# from jukebox.sample import sample_single_window, _sample, sample_partial_window, upsample, load_prompts
# from jukebox.utils.dist_utils import setup_dist_from_mpi
# from jukebox.utils.torch_utils import empty_cache

# # Setup device
# rank, local_rank, device = setup_dist_from_mpi()

# # Jukebox model path
# JUKBOX_PATH = "/home/paperspace/jukebox/"

# # Model selection
# model = "5b_lyrics"  # Options: ["5b_lyrics", "5b", "1b_lyrics"]

# # Define Hyperparams
# hps = Hyperparams()
# hps.sr = 44100
# hps.n_samples = 2
# hps.name = os.path.join(JUKBOX_PATH, "samples")
# hps.levels = 3
# hps.hop_fraction = [1, 1, 0.125]  # Faster upsampling

# # Artist, genre, and lyrics
# select_artist = "the beatles"
# select_genre = "pop rock"
# your_lyrics = """Enter your lyrics here"""

# # Desired sample length in seconds
# sample_length_in_seconds = 70
# if sample_length_in_seconds < 24:
    # sample_length_in_seconds = 24
    # print("Sample length too short, set to 24 seconds.")

# # Load models
# vqvae, *priors = MODELS[model]
# vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length=1048576)), device)
# top_prior = make_prior(setup_hparams(priors[-1], dict()), vqvae, device)

# # # Define sample length
# # hps.sample_length = (
    # # int(sample_length_in_seconds * hps.sr) // top_prior.raw_to_tokens
# # ) * top_prior.raw_to_tokens

# hps.sample_length = (
    # int(sample_length_in_seconds * hps.sr) // top_prior.raw_to_tokens
# ) * top_prior.raw_to_tokens

# assert hps.sample_length >= top_prior.n_ctx * top_prior.raw_to_tokens, \
    # f"Sample length is too short. Minimum required: {top_prior.n_ctx * top_prior.raw_to_tokens}"
    
# # # Ensure sample length is sufficient
# # assert hps.sample_length >= top_prior.n_ctx * top_prior.raw_to_tokens, \
    # # f"Sample length is too short. Minimum required: {top_prior.n_ctx * top_prior.raw_to_tokens}"

# # Metadata configuration
# metas = [
    # dict(
        # artist=select_artist,
        # genre=select_genre,
        # total_length=hps.sample_length,
        # offset=0,
        # lyrics=your_lyrics,
    # )
# ] * hps.n_samples

# # Labels
# labels = [None, None, top_prior.labeller.get_batch_labels(metas, device)]

# # Sampling settings
# chunk_size = 64 if model in ("5b", "5b_lyrics") else 128
# sampling_temperature = 0.98
# lower_level_chunk_size = 32
# sampling_kwargs = [
    # dict(temp=0.99, fp16=True, max_batch_size=8, chunk_size=lower_level_chunk_size),
    # dict(temp=0.99, fp16=True, max_batch_size=8, chunk_size=lower_level_chunk_size),
    # dict(temp=sampling_temperature, fp16=True, max_batch_size=8, chunk_size=chunk_size),
# ]

# # Mode of operation
# mode = "primed"  # Options: ["ancestral", "primed", "continue", "upsample"]

# # Prime mode settings
# audio_file = os.path.join(JUKBOX_PATH, "./Great.wav")
# prompt_length_in_seconds = 10
# print(">> prompt_length_in_seconds >>>>> ")

# sample_hps = Hyperparams(
    # dict(
        # mode=mode,
        # codes_file=None,
        # audio_file=audio_file,
        # prompt_length_in_seconds=prompt_length_in_seconds,
    # )
# )

# # Handle different modes
# if mode == "primed":
    # assert sample_hps.audio_file is not None, "Audio file required for primed mode."
    # duration = (
        # int(sample_hps.prompt_length_in_seconds * hps.sr) // top_prior.raw_to_tokens
    # ) * top_prior.raw_to_tokens
    # x = load_prompts([sample_hps.audio_file], duration, hps)
    # zs = top_prior.encode(x, start_level=0, end_level=len(priors), bs_chunks=x.shape[0])
    # zs = _sample(zs, labels, sampling_kwargs, [None, None, top_prior], [2], hps)
# else:
    # raise ValueError(f"Unknown sample mode {mode}")

# # Upsampling
# print(">> upsampling>>>>> ")
# del top_prior
# empty_cache()
# top_prior = None
# upsamplers = [
    # make_prior(setup_hparams(prior, dict()), vqvae, device)
    # for prior in priors[:-1]
# ]
# upsamplers = [make_prior(setup_hparams(prior, dict()), vqvae, device) for prior in priors[:-1]]
# zs = upsample(zs, labels, sampling_kwargs, upsamplers, hps)

# print("Sampling complete. Files saved in:", hps.name)

import os
import torch as t
from jukebox.make_models import make_vqvae, make_prior, MODELS
from jukebox.hparams import Hyperparams, setup_hparams
from jukebox.sample import sample_single_window, _sample, upsample, load_prompts
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.utils.torch_utils import empty_cache

# Setup device
rank, local_rank, device = setup_dist_from_mpi()

# Base path for Jukebox installation
JUKBOX_PATH = "/home/paperspace/jukebox/"

# Model selection
model = "5b_lyrics"  # Options: ["5b_lyrics", "5b", "1b_lyrics"]

# Define Hyperparams
hps = Hyperparams()
hps.sr = 44100
hps.n_samples = 2
hps.name = os.path.join(JUKBOX_PATH, "samples")  # Output folder
hps.levels = 3
hps.hop_fraction = [1, 1, 0.125]  # Faster upsampling

# User inputs: artist, genre, lyrics, and sample length
select_artist = "the beatles"
select_genre = "pop rock"
your_lyrics = """Enter your lyrics here"""
sample_length_in_seconds = 70  # Adjust this to your desired length

if sample_length_in_seconds < 24:
    sample_length_in_seconds = 24
    print("Sample length too short, set to 24 seconds.")

# Load models
vqvae_name, *prior_names = MODELS[model]
vqvae = make_vqvae(setup_hparams(vqvae_name, dict(sample_length=1048576)), device)
priors = [make_prior(setup_hparams(name, dict()), vqvae, device) for name in prior_names]

# Define sample length
hps.sample_length = (
    int(sample_length_in_seconds * hps.sr) // priors[-1].raw_to_tokens
) * priors[-1].raw_to_tokens

# Ensure sample length is sufficient
assert hps.sample_length >= priors[-1].n_ctx * priors[-1].raw_to_tokens, \
    f"Sample length is too short. Minimum required: {priors[-1].n_ctx * priors[-1].raw_to_tokens}"

# Metadata configuration
metas = [
    dict(
        artist=select_artist,
        genre=select_genre,
        total_length=hps.sample_length,
        offset=0,
        lyrics=your_lyrics,
    )
] * hps.n_samples

# Labels
labels = [None, None] + [prior.labeller.get_batch_labels(metas, device) for prior in priors[:-1]]

# Sampling settings
chunk_size = 64 if model in ("5b", "5b_lyrics") else 128
sampling_temperature = 0.98
lower_level_chunk_size = 32
sampling_kwargs = [
    dict(temp=0.99, fp16=True, max_batch_size=8, chunk_size=lower_level_chunk_size),
    dict(temp=0.99, fp16=True, max_batch_size=8, chunk_size=lower_level_chunk_size),
    dict(temp=sampling_temperature, fp16=True, max_batch_size=8, chunk_size=chunk_size),
]

# Mode of operation
mode = "primed"  # Options: ["ancestral", "primed", "continue", "upsample"]
audio_file = os.path.join(JUKBOX_PATH, "/Great.wav") # Path to prompt audio
prompt_length_in_seconds = 10

sample_hps = Hyperparams(
    dict(
        mode=mode,
        codes_file=None,
        audio_file=audio_file,
        prompt_length_in_seconds=prompt_length_in_seconds,
    )
)

# Handle different modes
if mode == "primed":
    assert sample_hps.audio_file is not None, "Audio file required for primed mode."
    duration = (
        int(sample_hps.prompt_length_in_seconds * hps.sr) // priors[-1].raw_to_tokens
    ) * priors[-1].raw_to_tokens
    x = load_prompts([sample_hps.audio_file], duration, hps)
    zs = priors[-1].encode(x, start_level=0, end_level=len(priors), bs_chunks=x.shape[0])
elif mode == "ancestral":
    zs = [t.zeros(hps.n_samples, 0, dtype=t.long, device=device) for _ in priors]
elif mode == "upsample":
    assert sample_hps.codes_file is not None, "Codes file required for upsampling mode."
    data = t.load(sample_hps.codes_file, map_location="cpu")
    zs = [z.cpu() for z in data["zs"]]
elif mode == "continue":
    data = t.load(sample_hps.codes_file, map_location="cpu")
    zs = [z.cuda() for z in data["zs"]]
else:
    raise ValueError(f"Unknown sample mode {mode}")

# Upsampling
del priors[-1]
empty_cache()
upsamplers = priors[:-1]
zs = upsample(zs, labels, sampling_kwargs, upsamplers, hps)

print("Sampling complete. Files saved in:", hps.name)
