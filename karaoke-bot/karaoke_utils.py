import os
import subprocess
import torch
import torchaudio
import soundfile as sf
from demucs import pretrained
from demucs.apply import apply_model
from transformers import pipeline
import pysubs2
from pathlib import Path
import gc
import shutil

def prepare_audio(input_path, output_audio="temp_audio.wav"):
    input_path = str(input_path)
    output_audio = Path(output_audio)
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    cmd_audio = [
        "ffmpeg", "-i", input_path,
        "-af", "loudnorm=I=-16:LRA=11:TP=-1.5, silenceremove=1:0:-50dB",
        "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
        "-y", str(output_audio)
    ]
    subprocess.run(cmd_audio, check=True, stderr=subprocess.DEVNULL)
    return str(output_audio)

def separate_audio(input_path, device="cuda"):
    model = pretrained.get_model('htdemucs')
    model.to(device).eval()
    wav, orig_sr = torchaudio.load(input_path)
    target_sr = model.samplerate
    if orig_sr != target_sr:
        wav = torchaudio.transforms.Resample(orig_sr, target_sr)(wav)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    wav = wav.to(device).unsqueeze(0)
    with torch.inference_mode():
        sources = apply_model(model, wav, device=device)[0]
    vocal = sources[3].cpu().numpy().T
    instr = (sources[0] + sources[1] + sources[2]).cpu().numpy().T
    os.makedirs("separated", exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    vocal_path = f"separated/{base}_vocals.wav"
    instr_path = f"separated/{base}_instrumental.wav"
    sf.write(vocal_path, vocal, target_sr)
    sf.write(instr_path, instr, target_sr)
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return vocal_path, instr_path

def preprocess_audio(input_path, output_path="optimized.wav"):
    wav, sr = torchaudio.load(input_path)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    torchaudio.save(output_path, wav, 16000)
    return str(output_path)

def transcribe_audio(audio_path, model_path="openai/whisper-large-v3"):
    audio_path = str(audio_path)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        device="cuda",
        generate_kwargs={"language": "en", "task": "transcribe"}
    )
    result = pipe(audio_path, return_timestamps="word")
    words = []
    for chunk in result.get("chunks", []):
        if chunk["timestamp"][0] is not None and chunk["timestamp"][1] is not None:
            words.append({
                "word": chunk["text"].strip(),
                "start": chunk["timestamp"][0],
                "end": chunk["timestamp"][1]
            })
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    return words, result.get("text", "")

def words_to_ass_karaoke(words, output_ass="karaoke.ass", resolution="1280x720", words_per_line=5, min_pause=0.15):
    valid_words = [w for w in words if w["start"] is not None and w["end"] is not None]
    if not valid_words:
        subs = pysubs2.SSAFile()
        subs.save(output_ass)
        return output_ass

    phrases = []
    cur = {"words": [], "start": valid_words[0]["start"], "word_objects": []}
    for i, w in enumerate(valid_words):
        if i == 0:
            cur["word_objects"].append(w)
            cur["words"].append(w["word"])
            continue
        prev_word = valid_words[i-1]["word"]
        is_new_sentence = w["word"][0].isupper() and (
            prev_word.endswith('.') or prev_word.endswith('!') or prev_word.endswith('?')
        )
        pause = w["start"] - valid_words[i-1]["end"]
        if pause > min_pause or len(cur["word_objects"]) >= words_per_line or is_new_sentence:
            cur["end"] = valid_words[i-1]["end"]
            phrases.append(cur)
            cur = {"word_objects": [w], "words": [w["word"]], "start": w["start"]}
        else:
            cur["word_objects"].append(w)
            cur["words"].append(w["word"])
    if cur["word_objects"]:
        cur["end"] = valid_words[-1]["end"]
        phrases.append(cur)

    subs = pysubs2.SSAFile()
    width, height = map(int, resolution.split('x'))
    subs.info['PlayResX'] = width
    subs.info['PlayResY'] = height

    style = pysubs2.SSAStyle()
    style.fontname = "Arial"
    style.fontsize = 100
    style.primarycolor = pysubs2.Color(255, 215, 0)
    style.secondarycolor = pysubs2.Color(170, 170, 170)
    style.outlinecolor = pysubs2.Color(0, 0, 0)
    style.backcolor = pysubs2.Color(0, 0, 0)
    style.bold = True
    style.borderstyle = 3
    style.outline = 4
    style.shadow = 4
    style.alignment = 5
    style.margin_l = 0
    style.margin_r = 0
    style.margin_v = 0
    style.encoding = 1
    subs.styles["Karaoke"] = style

    for phrase in phrases:
        start = phrase["start"]
        end = phrase["end"]
        text_parts = []
        for w in phrase["word_objects"]:
            duration = w["end"] - w["start"]
            k_value = max(1, int(duration * 100))
            text_parts.append(f"{{\\k{k_value}}}{w['word']}")
        full_text = " ".join(text_parts)
        if len(full_text) > 60:
            parts = full_text.split()
            if len(parts) > 2:
                mid = len(parts) // 2
                line1 = " ".join(parts[:mid])
                line2 = " ".join(parts[mid:])
                full_text = line1 + "\\N" + line2
        event = pysubs2.SSAEvent(
            start=int(start * 1000),
            end=int(end * 1000),
            style="Karaoke",
            text=full_text
        )
        subs.append(event)

    subs.save(output_ass)
    return output_ass

def create_karaoke_video(instrumental_audio, ass_file, output_video="karaoke_output.mp4", resolution="1280x720"):
    instrumental_audio = str(instrumental_audio)
    ass_file = str(ass_file)
    output_video = str(output_video)
    cmd_probe = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", instrumental_audio]
    duration = float(subprocess.check_output(cmd_probe).strip())
    cmd = ["ffmpeg", "-y",
           "-f", "lavfi", "-i", f"color=c=black:s={resolution}:d={duration}",
           "-i", instrumental_audio,
           "-vf", f"ass={ass_file}",
           "-c:v", "libx264", "-c:a", "aac",
           "-shortest", output_video]
    subprocess.run(cmd, check=True)
    return output_video

def cleanup_temp_files(*paths):
    for p in paths:
        if p and os.path.exists(p):
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p)
                else:
                    os.remove(p)
            except Exception:
                pass
