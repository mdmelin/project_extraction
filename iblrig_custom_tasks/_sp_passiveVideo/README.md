# Passive video protocol
Presents a fullscreen video n times in a row.

## Workflow

1. Protocol spacer outputted
2. Start trial (record timestamp)
3. Playback of `VIDEO` initiated
3. Bpod output HIGH
4. Await video end
5. Bpod output LOW
6. End trial (record timestamp)
7. Wait for `ITI_DALAY_SECS`
8. Repeat `NREPEATS` times

## Setup

1. Install VLC media player for Windows at https://www.videolan.org/vlc/
2. `pip install -r ./requirements.txt`
