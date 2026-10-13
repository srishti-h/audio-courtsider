# Data

Nothing in this folder is committed. `make data` downloads and builds everything:

| Source | What | License / access |
|---|---|---|
| [SoccerNet](https://www.soccer-net.org/) action-spotting labels (`Labels-v2.json`) | 110k timestamped events, 500 matches | public labels via the `SoccerNet` pip package |
| [SoccerNet-Echoes](https://github.com/SoccerNet/sn-echoes) | Whisper transcripts of the broadcast commentary, video-aligned | CC BY 4.0 |
| [football-data.co.uk](https://www.football-data.co.uk/) | results + Pinnacle/Bet365 odds, 5 leagues, 2012-17 | free for research |

The raw SoccerNet broadcast video/audio is gated: request access on
[SoccerNet/SoccerNet_raw_HQ](https://huggingface.co/datasets/SoccerNet/SoccerNet_raw_HQ) (NDA
accepted on Hugging Face), then `hf auth login`. Until then the audio components run on synthetic
commentary (`courtsider.synth`); with access, `python -m courtsider.eval.run_real_audio` runs the
same pipeline on real broadcast audio.
